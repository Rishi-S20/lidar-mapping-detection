"""Dark-safe live 3D mapping: IMU orientation + LiDAR ICP position + voxel map (ADR-0015).

Per frame:
  1. depth -> camera points (confidence and range filtered)
  2. orientation guess = previous pose rotation x IMU rotation change since the last frame
     (gyro-driven, no camera); position guess = previous position + last velocity
  3. refine both with point-to-plane ICP against the local map
  4. accept if the fit is good and agrees with the IMU; then add the points to the map
     Otherwise keep the IMU orientation, hold the position, and don't add points ("lost").

`pose_source="arkit"` uses ARKit's camera-based pose instead. It is for
comparison with the lights on only; it fails in the dark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (CAM_TO_DEVICE, arkit_pose_to_world, backproject, remove_edge_pixels, remove_grazing_pixels,
                       rotation_angle_deg, world_from_device)
from .icp import estimate_normals, icp_point_to_plane, voxel_downsample
from .protocol import DepthFrame


_OFF = 1 << 20  # voxel index offset; supports +-1M voxels (+-30 km at 3 cm)


def _pack(k: np.ndarray) -> np.ndarray:
    k = k + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


class VoxelMap:
    """Sparse voxel grid: running mean position and hit count per voxel (packed int64 keys)."""

    def __init__(self, voxel: float = 0.02) -> None:
        self.voxel = voxel
        self.keys = np.zeros(0, np.int64)
        self.sums = np.zeros((0, 3))
        self.counts = np.zeros(0, np.int64)
        self.misses = np.zeros(0, np.int64)  # times a depth ray passed through this voxel (free space)
        self.carving = False  # set by Mapper; when on, points() drops voxels seen through more than hit

    def __len__(self) -> int:
        return len(self.counts)

    def add(self, points: np.ndarray, counts: np.ndarray | None = None) -> None:
        """Add points (each counts once), or voxel means with their hit counts (map merging)."""
        if len(points) == 0:
            return
        w = np.ones(len(points)) if counts is None else np.asarray(counts, float)
        k = _pack(np.floor(points / self.voxel).astype(np.int64))
        uk, inv = np.unique(k, return_inverse=True)
        s = np.zeros((len(uk), 3))
        np.add.at(s, inv, points * w[:, None])
        c = np.bincount(inv, weights=w, minlength=len(uk)).astype(np.int64)
        pos = np.searchsorted(self.keys, uk)
        hit = (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == uk) if len(self.keys) else np.zeros(len(uk), bool)
        self.sums[pos[hit]] += s[hit]
        self.counts[pos[hit]] += c[hit]
        new = ~hit
        if new.any():
            keys = np.concatenate([self.keys, uk[new]])
            order = np.argsort(keys, kind="stable")
            self.keys = keys[order]
            self.sums = np.concatenate([self.sums, s[new]])[order]
            self.counts = np.concatenate([self.counts, c[new]])[order]
            self.misses = np.concatenate([self.misses, np.zeros(int(new.sum()), np.int64)])[order]

    def carve(self, origin: np.ndarray, ends: np.ndarray, stop_before: float = 0.08, start: float = 0.25) -> None:
        """Free-space carving: every voxel a ray crosses before its hit is evidence of empty space."""
        if len(ends) == 0 or len(self.keys) == 0:
            return
        d = ends - origin
        L = np.linalg.norm(d, axis=1)
        ok = L > start + stop_before
        d, L = d[ok] / L[ok, None], L[ok]
        steps = np.arange(start, L.max() - stop_before, self.voxel)
        if len(steps) == 0:
            return
        pts = origin + d[:, None, :] * steps[None, :, None]  # [rays, steps, 3]
        valid = steps[None, :] < (L - stop_before)[:, None]
        k = _pack(np.floor(pts[valid] / self.voxel).astype(np.int64))
        k = np.unique(k)  # count each voxel at most once per frame
        pos = np.searchsorted(self.keys, k)
        pos = np.minimum(pos, len(self.keys) - 1)
        hit = self.keys[pos] == k
        self.misses[pos[hit]] += 1

    def points(self, min_count: int = 1) -> tuple[np.ndarray, np.ndarray]:
        m = self.counts >= min_count
        if self.carving:
            # occupancy: keep voxels hit more often than they were seen through (free space)
            m &= self.misses <= self.counts
        return self.sums[m] / self.counts[m, None], self.counts[m]

    def sample(self, n: int, min_count: int = 2, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        p, c = self.points(min_count)
        if len(p) > n:
            idx = (rng or np.random.default_rng(0)).choice(len(p), n, replace=False)
            p, c = p[idx], c[idx]
        return p, c


def _tilt_error_deg(R: np.ndarray, R_ref: np.ndarray) -> float:
    """Angle between where R and R_ref say "world up" points in the camera frame (roll/pitch only)."""
    up = np.array([0.0, 0.0, 1.0])
    a, b = R.T @ up, R_ref.T @ up
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def _match_tilt(R: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """R with its roll/pitch replaced by R_ref's (keeps R's heading). Both world <- camera."""
    up = np.array([0.0, 0.0, 1.0])
    a, b = R.T @ up, R_ref.T @ up  # world-up in camera coords, per rotation
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return R
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Q = np.eye(3) + vx + vx @ vx / (1 + c)  # camera-frame rotation taking a -> b
    return R @ Q.T


def build_target(points: np.ndarray, voxel: float, max_points: int, rng: np.random.Generator):
    """Downsampled points + normals + KD-tree, for ICP. None if too few planar points."""
    p = voxel_downsample(points, voxel)
    if len(p) > max_points:
        p = p[rng.choice(len(p), max_points, replace=False)]
    if len(p) < 100:
        return None
    n, planarity = estimate_normals(p, cKDTree(p))
    keep = planarity > 0.3
    p, n = p[keep], n[keep]
    return (p, n, cKDTree(p)) if len(p) >= 100 else None


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def global_search(src: np.ndarray, R_base: np.ndarray, target, positions: list[np.ndarray],
                  yaw_step_deg: float = 30, keep: int = 6, yaws_deg=None):
    """Align `src` (already gravity-levelled by R_base) to `target` with unknown yaw and position.

    Candidates: yaw every `yaw_step_deg` x `positions`, scored by the fraction of points within
    15 cm of the target; the best `keep` are refined with ICP. Returns the best ICPResult.
    """
    tgt, tgt_n, tree = target
    cands = []
    for yaw in np.radians(np.arange(0, 360, yaw_step_deg) if yaws_deg is None else yaws_deg):
        R0 = _rotz(yaw) @ R_base
        moved = src @ R0.T
        for p0 in positions:
            d, _ = tree.query(moved + p0, distance_upper_bound=0.15)
            cands.append((float(np.isfinite(d).mean()), R0, p0))
    cands.sort(key=lambda x: -x[0])
    best = None
    for _, R0, p0 in cands[:keep]:
        res = icp_point_to_plane(src, tgt, tgt_n, tree, R0, p0, max_iter=20, max_dist=0.4)
        if best is None or res.inlier_frac > best.inlier_frac:
            best = res
    return best


def spread_positions(trajectory: list[np.ndarray], first: np.ndarray, n: int = 9, gap: float = 0.4):
    out = [first.copy()]
    for q in trajectory[::-1]:
        if len(out) >= n:
            break
        if min(np.linalg.norm(q - p) for p in out) > gap:
            out.append(q.copy())
    return out


@dataclass
class Segment:
    """A finished (or set-aside) map that the current one may later merge into."""
    map: VoxelMap
    trajectory: list


@dataclass
class FrameResult:
    seq: int
    status: str  # "waiting", "init", "tracking", "relocalized", "merged", "lost", "arkit"
    R: np.ndarray  # world <- camera (OpenCV)
    t: np.ndarray
    n_points: int
    valid_fraction: float
    rmse: float = float("nan")
    inlier_frac: float = float("nan")
    gravity_err_deg: float = float("nan")
    imu_disagreement_deg: float = float("nan")


class Mapper:
    def __init__(self, voxel: float = 0.02, icp_voxel: float = 0.04, max_range: float = 4.5, min_conf: int = 1,
                 pose_source: str = "imu_icp", stride: int = 1, max_target: int = 15000, map_stride: int = 1,
                 target_every: int = 3, max_source: int = 3000,
                 rot_prior_weight: float = 0.02, max_imu_disagreement_deg: float = 10.0,
                 max_rmse: float = 0.04, min_inlier_frac: float = 0.2, min_init_points: int = 3000,
                 min_matches: int = 300, map_max_range: float = 3.5, edge_filter: float | None = 0.05,
                 tilt_prior_weight: float = 5.0, integrate_max_rmse: float | None = None,
                 integrate_max_tilt_deg: float | None = None, grazing_min_cos: float | None = 0.25,
                 carve_rays: int = 2000, carve_every: int = 2) -> None:
        if pose_source not in ("imu_icp", "arkit"):
            raise ValueError(pose_source)
        self.map = VoxelMap(voxel)
        self.icp_voxel, self.max_range, self.min_conf = icp_voxel, max_range, min_conf
        self.pose_source, self.stride, self.max_target = pose_source, stride, max_target
        self.rot_prior_weight = rot_prior_weight
        self.max_imu_disagreement_deg, self.max_rmse, self.min_inlier_frac = (
            max_imu_disagreement_deg, max_rmse, min_inlier_frac)
        self.R: np.ndarray | None = None
        self.t = np.zeros(3)
        self.v = np.zeros(3)
        self.prev_imu: np.ndarray | None = None
        self.prev_time: float | None = None
        self.trajectory: list[np.ndarray] = []
        self.lost_streak = 0
        self.rng = np.random.default_rng(0)
        self.map_stride, self.target_every, self.max_source = map_stride, target_every, max_source
        self.min_init_points = min_init_points
        self.min_map_voxels = 2000
        self.resets = 0
        self._tgt = None
        self._tgt_age = 0
        self._tgt_center = np.zeros(3)
        self.min_matches = min_matches
        self.map_max_range, self.edge_filter = map_max_range, edge_filter
        # tilt held to gravity in ICP; frames whose fit or tilt is worse than these are tracked but not mapped
        self.tilt_prior_weight = tilt_prior_weight
        self.integrate_max_rmse, self.integrate_max_tilt_deg = integrate_max_rmse, integrate_max_tilt_deg
        self.skipped_integrations = 0
        self.grazing_min_cos, self.carve_rays, self.carve_every = grazing_min_cos, carve_rays, carve_every
        self._integrations = 0
        self.map.carving = carve_rays > 0
        self._map_pts = None  # points of the current frame that go into the map
        self.last_seq = -1
        self.need_reloc = False
        self.reloc_after, self.reloc_every = 6, 4
        self.relocalizations = 0
        self._reloc_tgt = None
        self._reloc_tgt_size = 0
        self.reloc_give_up = 4  # failed relocalization attempts before starting a new segment
        self.reloc_attempts = 0
        self.segments: list[Segment] = []  # older maps not (yet) merged with the current one
        self.merge_every, self.min_merge_voxels = 15, 15000
        self.merges = 0
        self._frames_since_merge_try = 0

    # ------------------------------------------------------------------ helpers
    def _target(self, center: np.ndarray | None, voxel: float | None = None):
        p, _ = self.map.points(min_count=1)
        if len(p) == 0:
            return None
        if center is not None:
            p = p[np.linalg.norm(p - center, axis=1) <= self.max_range + 1.0]
        p = voxel_downsample(p, voxel or self.icp_voxel * 0.75)
        if len(p) > self.max_target:
            p = p[self.rng.choice(len(p), self.max_target, replace=False)]
        if len(p) < 100:
            return None
        tree = cKDTree(p)
        n, planarity = estimate_normals(p, tree)
        keep = planarity > 0.3
        p, n = p[keep], n[keep]
        if len(p) < 100:
            return None
        return p, n, cKDTree(p)

    # ------------------------------------------------------------------ main step
    def process(self, f: DepthFrame) -> FrameResult:
        try:
            return self._process(f)
        finally:
            self.prev_time = f.t

    def _process(self, f: DepthFrame) -> FrameResult:
        pts = backproject(f.depth, f.K, f.confidence, self.min_conf, max_range=self.max_range, stride=self.stride)
        # The map gets a cleaner subset: no edge "flying pixels", and only the precise short range.
        # Tracking (ICP) keeps the full range, which gives it more geometry to lock onto.
        clean = remove_edge_pixels(f.depth, self.edge_filter) if self.edge_filter else f.depth
        if self.grazing_min_cos:
            clean = remove_grazing_pixels(clean, f.K, self.grazing_min_cos)
        self._map_pts = backproject(clean, f.K, f.confidence, self.min_conf, max_range=self.map_max_range,
                                    stride=self.stride)
        R_wd, gerr = world_from_device(f.attitude, f.gravity)
        R_imu = R_wd @ CAM_TO_DEVICE  # world <- camera, from IMU alone
        dt = (f.t - self.prev_time) if self.prev_time is not None else 0.0

        if self.pose_source == "arkit":
            if f.arkit_pose is None:
                raise ValueError("pose_source='arkit' but the frame has no arkit_pose")
            R, t = arkit_pose_to_world(f.arkit_pose)
            self._accept(R, t, pts, dt)
            return FrameResult(f.seq, "arkit", R, t, len(pts), f.valid_fraction, gravity_err_deg=gerr)

        if self.R is None:  # first usable frame defines the map; origin at the phone, z up from gravity
            extent = float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0
            if len(pts) < self.min_init_points or extent < 1.0 or float(np.median(pts[:, 2])) < 0.5:
                # First phone run (2026-11-14): the LiDAR sent empty frames for ~1 s, then frames of a
                # surface 12-22 cm away (desk/hand). A map started from that can never match the room.
                # Wait for a real view: enough points, >= 1 m across, median depth >= 0.5 m.
                return FrameResult(f.seq, "waiting", R_imu, np.zeros(3), len(pts), f.valid_fraction,
                                   gravity_err_deg=gerr)
            self._accept(R_imu, np.zeros(3), pts, dt)
            self.prev_imu = R_imu
            return FrameResult(f.seq, "init", R_imu, np.zeros(3), len(pts), f.valid_fraction, gravity_err_deg=gerr)

        # New phone session (app restarted streaming) or a long gap: the IMU's yaw reference has
        # reset to a new arbitrary direction and the phone may have moved. Find it again.
        if f.seq < self.last_seq or dt > 1.5:
            self.need_reloc, self.prev_imu, self.v = True, None, np.zeros(3)
        self.last_seq = f.seq

        if self.need_reloc or self.lost_streak >= self.reloc_after:
            due = self.need_reloc and self.lost_streak == 0 or self.lost_streak % self.reloc_every == 0
            if due:
                res = self._relocalize(pts, R_imu, yaw_known=f.attitude_ref == "magnetic_north")
                if res is not None:
                    self.need_reloc, self.lost_streak, self.prev_imu = False, 0, R_imu
                    self.reloc_attempts = 0
                    self.relocalizations += 1
                    self._tgt = None
                    self._accept(res.R, res.t, pts, 0.0)
                    return FrameResult(f.seq, "relocalized", res.R, res.t, len(pts), f.valid_fraction, res.rmse,
                                       res.inlier_frac, gerr)
                self.reloc_attempts += 1
                if self.reloc_attempts >= self.reloc_give_up:
                    self._start_segment()
                    return self._process(f)  # becomes the first frame of the new segment
            if self.need_reloc:
                self.prev_imu = R_imu
                return self._lost(f, R_imu, len(pts), gerr)

        # IMU rotation change since the last frame, applied to the last refined rotation
        R_guess = (R_imu @ self.prev_imu.T) @ self.R if self.prev_imu is not None else R_imu
        t_guess = self.t + self.v * dt
        self.prev_imu = R_imu

        if (self._tgt is None or self._tgt_age >= self.target_every
                or np.linalg.norm(t_guess - self._tgt_center) > 0.3):
            self._tgt, self._tgt_age, self._tgt_center = self._target(t_guess), 0, t_guess.copy()
        self._tgt_age += 1
        target = self._tgt
        if target is None and len(self.map) < self.min_map_voxels and self.lost_streak >= 5:
            # The map never got going (too small to align against): start over from this frame.
            self.reset()
            return self._process(f)
        src = voxel_downsample(pts, self.icp_voxel)
        if len(src) > self.max_source:
            src = src[self.rng.choice(len(src), self.max_source, replace=False)]
        if target is None or len(src) < 100:
            return self._lost(f, R_guess, len(pts), gerr)
        tgt, tgt_n, tree = target
        # roll/pitch prior: the IMU's absolute tilt (gravity), not the propagated guess
        R_prior = R_guess
        if self.tilt_prior_weight > 0:
            R_prior = _match_tilt(R_guess, R_imu)
        res = icp_point_to_plane(src, tgt, tgt_n, tree, R_guess, t_guess,
                                 rot_prior=R_prior, rot_prior_weight=self.rot_prior_weight,
                                 tilt_prior_weight=self.tilt_prior_weight)
        disagree = rotation_angle_deg(res.R @ R_guess.T)
        ok = (res.inlier_frac >= self.min_inlier_frac and res.inlier_frac * len(src) >= self.min_matches
              and res.rmse <= self.max_rmse and disagree <= self.max_imu_disagreement_deg)
        if not ok:
            r = self._lost(f, R_guess, len(pts), gerr)
            r.rmse, r.inlier_frac, r.imu_disagreement_deg = res.rmse, res.inlier_frac, disagree
            return r
        self.lost_streak = 0
        self.reloc_attempts = 0
        tilt_err = _tilt_error_deg(res.R, R_imu)
        integrate = ((self.integrate_max_rmse is None or res.rmse <= self.integrate_max_rmse)
                     and (self.integrate_max_tilt_deg is None or tilt_err <= self.integrate_max_tilt_deg))
        if not integrate:
            self.skipped_integrations += 1
        self._accept(res.R, res.t, pts, dt, integrate=integrate)
        status = "tracking"
        self._frames_since_merge_try += 1
        if self.segments and self._frames_since_merge_try >= self.merge_every:
            self._frames_since_merge_try = 0
            if self._try_merge():
                status = "merged"
        return FrameResult(f.seq, status, self.R, self.t, len(pts), f.valid_fraction, res.rmse,
                           res.inlier_frac, gerr, disagree)

    def _relocalize(self, pts: np.ndarray, R_imu: np.ndarray, yaw_known: bool = False):
        """Global search when lost: roll/pitch from gravity are trusted, yaw and position are not."""
        if self._reloc_tgt is None or len(self.map) > 1.1 * self._reloc_tgt_size:
            p, _ = self.map.points(1)
            self._reloc_tgt, self._reloc_tgt_size = build_target(p, 0.06, self.max_target, self.rng), len(self.map)
        if self._reloc_tgt is None:
            return None
        src = voxel_downsample(pts, 0.06)
        if len(src) < 150:
            return None
        if len(src) > 2000:
            src = src[self.rng.choice(len(src), 2000, replace=False)]
        positions = spread_positions(self.trajectory, self.t, n=16, gap=0.3)
        best = None
        if yaw_known:  # magnetic-north heading: only a small heading error to absorb
            best = global_search(src, R_imu, self._reloc_tgt, positions, yaws_deg=[-20, -10, 0, 10, 20])
        if best is None or best.inlier_frac < 0.45:
            alt = global_search(src, R_imu, self._reloc_tgt, positions)
            if best is None or (alt is not None and alt.inlier_frac > best.inlier_frac):
                best = alt
        if best is not None and best.inlier_frac >= 0.45 and best.rmse <= 0.035:
            return best
        return None

    def _start_segment(self) -> None:
        """Relocalization keeps failing (new area or long gap): set the current map aside, start fresh."""
        if len(self.map) >= self.min_map_voxels:
            self.segments.append(Segment(self.map, self.trajectory))
        carving = self.map.carving
        self.map = VoxelMap(self.map.voxel)
        self.map.carving = carving
        self.trajectory = []
        self.R, self.t, self.v, self.prev_imu = None, np.zeros(3), np.zeros(3), None
        self._tgt, self._reloc_tgt, self._reloc_tgt_size = None, None, 0
        self.need_reloc, self.lost_streak, self.reloc_attempts = False, 0, 0
        self._frames_since_merge_try = 0

    def _try_merge(self) -> bool:
        """Align the whole current map to each set-aside segment; on success, fold it in."""
        if not self.segments or len(self.map) < self.min_merge_voxels:
            return False
        p, c = self.map.points(1)
        src = voxel_downsample(p[c >= 2], 0.08)
        if len(src) > 3000:
            src = src[self.rng.choice(len(src), 3000, replace=False)]
        for i, seg in sorted(enumerate(self.segments), key=lambda x: -len(x[1].map)):
            sp, _ = seg.map.points(1)
            target = build_target(sp, 0.06, self.max_target, self.rng)
            if target is None:
                continue
            # the current map's origin is where this segment started; try it near past positions
            res = global_search(src, np.eye(3), target, spread_positions(seg.trajectory, seg.trajectory[-1], n=12))
            if res is None or res.inlier_frac < 0.4 or res.rmse > 0.035:
                continue
            R, t = res.R, res.t
            mp, mc = self.map.points(1)
            seg.map.add(mp @ R.T + t, mc)
            self.trajectory = seg.trajectory + [R @ q + t for q in self.trajectory]
            self.map = seg.map
            if self.R is not None:
                self.R, self.t, self.v = R @ self.R, R @ self.t + t, R @ self.v
            del self.segments[i]
            self._tgt, self._reloc_tgt, self._reloc_tgt_size = None, None, 0
            self.merges += 1
            return True
        return False

    def reset(self) -> None:
        carving = self.map.carving
        self.map = VoxelMap(self.map.voxel)
        self.map.carving = carving
        self.trajectory = []
        self.R, self.t, self.v, self.prev_imu = None, np.zeros(3), np.zeros(3), None
        self._tgt, self._tgt_age = None, 0
        self.lost_streak = 0
        self.resets += 1

    def _accept(self, R, t, pts, dt, integrate: bool = True) -> None:
        if dt > 0:
            self.v = 0.5 * self.v + 0.5 * (t - self.t) / dt
        self.R, self.t = R, t
        self.trajectory.append(t.copy())
        if integrate:
            mp = self._map_pts if self._map_pts is not None else pts
            world = mp[::self.map_stride] @ R.T + t
            self.map.add(world)
            self._integrations += 1
            if self.carve_rays and len(world) and self._integrations % self.carve_every == 0:
                n = min(self.carve_rays, len(world))
                self.map.carve(t, world[self.rng.choice(len(world), n, replace=False)])

    def _lost(self, f, R_guess, n, gerr) -> FrameResult:
        self.lost_streak += 1
        self.R = R_guess  # keep orientation from the IMU; hold position
        self.v = np.zeros(3)
        return FrameResult(f.seq, "lost", R_guess, self.t.copy(), n, f.valid_fraction, gravity_err_deg=gerr)


def save_ply(path: str, points: np.ndarray, counts: np.ndarray | None = None) -> None:
    """ASCII PLY of the map (opens in MeshLab, CloudCompare, Blender)."""
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n")
        if counts is not None:
            fh.write("property int hits\n")
        fh.write("end_header\n")
        for i, p in enumerate(points):
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" + (f" {int(counts[i])}\n" if counts is not None else "\n"))


def floor_check(points_world: np.ndarray) -> dict:
    """Fit the dominant horizontal-ish plane below the sensor; report its tilt from z (sanity check)."""
    if len(points_world) < 200:
        return {"ok": False, "reason": "too few points"}
    z0 = np.percentile(points_world[:, 2], 2)
    low = points_world[points_world[:, 2] < z0 + 0.08]  # the lowest band: floor, not low furniture
    c = low - low.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    n = vt[-1]
    tilt = math.degrees(math.acos(abs(float(n[2]))))
    return {"ok": tilt < 10, "floor_tilt_deg": round(tilt, 2), "floor_z": round(float(low[:, 2].mean()), 3)}
