"""Synthetic depth stream: a furnished room rendered as LiDAR depth frames with IMU readings.

Lets the mapper, viewer link, and tests run without a phone. Only geometry and
the IMU are simulated (it's "dark": no camera data at all). Noise is a rough
guess at iPhone LiDAR behaviour (~1% of range, random dropouts), not measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import CAM_TO_DEVICE, DOWN, R_to_quat
from .protocol import DepthFrame

W, H = 256, 192
K_DEFAULT = np.array([[211.0, 0, 127.5], [0, 211.0, 95.5], [0, 0, 1]])  # ~ iPhone Pro ARKit depth intrinsics


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    inside: bool = False  # True = room shell (ray starts inside, hits the far wall)


def default_room() -> list[Box]:
    """5 x 4 x 2.6 m room, floor at z = -1.3 (phone held ~1.3 m up at the origin), some furniture."""
    z0 = -1.3
    return [
        Box(np.array([-1.5, -2.0, z0]), np.array([3.5, 2.0, z0 + 2.6]), inside=True),
        Box(np.array([1.8, -1.9, z0]), np.array([2.6, -1.1, z0 + 0.75])),  # table
        Box(np.array([0.6, 1.2, z0]), np.array([1.4, 1.9, z0 + 0.9])),  # armchair
        Box(np.array([3.0, -0.6, z0]), np.array([3.45, 0.9, z0 + 1.8])),  # bookcase
        Box(np.array([1.2, -0.4, z0]), np.array([2.0, 0.2, z0 + 0.25])),  # person-sized lump on the floor
    ]


def _ray_box(o, d, b: Box):
    inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t1 = (b.lo - o) * inv
    t2 = (b.hi - o) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    if b.inside:
        return np.where(tmax > 0, tmax, np.inf)
    hit = (tmax >= np.maximum(tmin, 0))
    return np.where(hit & (tmin > 0), tmin, np.inf)


def render_depth(R_wc: np.ndarray, t: np.ndarray, boxes: list[Box], K: np.ndarray = K_DEFAULT,
                 rng: np.random.Generator | None = None, noise: float = 0.01, dropout: float = 0.03,
                 max_range: float = 5.0) -> np.ndarray:
    v, u = np.mgrid[0:H, 0:W]
    rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    d_w = rays_c @ R_wc.T
    o = np.broadcast_to(t, d_w.shape)
    tt = np.full(len(d_w), np.inf)
    for b in boxes:
        tt = np.minimum(tt, _ray_box(o, d_w, b))
    depth = tt  # rays_c has z = 1, so the ray parameter is the z-depth
    rng = rng or np.random.default_rng(0)
    depth = depth * (1 + rng.normal(0, noise, depth.shape))
    depth[(rng.random(depth.shape) < dropout) | (depth > max_range) | ~np.isfinite(depth)] = np.nan
    return depth.reshape(H, W).astype(np.float32)


def device_rotation(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Phone held upright in portrait, rear camera looking along world +x rotated by yaw; pitch tilts it down."""
    # device axes in world for yaw 0: x_d = -y_w, y_d = +z_w, z_d = -x_w
    R0 = np.array([[0.0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    p = -math.radians(pitch_deg)  # positive pitch = camera looks down
    # pitching the camera down = rotating about the device x axis
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    y = math.radians(yaw_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ R0 @ Rx


def walk(n: int = 60, fps: float = 10.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """A slow scan: turn left/right while stepping ~1.2 m forward. Returns (R_world_device, position)."""
    out = []
    for i in range(n):
        s = i / max(n - 1, 1)
        yaw = 50 * math.sin(2 * math.pi * s)
        pitch = 15 + 10 * math.sin(4 * math.pi * s)
        pos = np.array([1.2 * s, 0.3 * math.sin(math.pi * s), 0.05 * math.sin(6 * math.pi * s)])
        out.append((device_rotation(yaw, pitch), pos))
    return out


def stream(n: int = 60, fps: float = 10.0, boxes: list[Box] | None = None, seed: int = 0,
           imu_yaw_drift_deg_per_s: float = 0.5, attitude_convention: str = "device_to_world"):
    """Yield DepthFrames for `walk(n)`. The IMU attitude drifts slowly in yaw, like a real gyro."""
    boxes = boxes or default_room()
    rng = np.random.default_rng(seed)
    for i, (R_wd, pos) in enumerate(walk(n, fps)):
        t = i / fps
        R_wc = R_wd @ CAM_TO_DEVICE
        depth = render_depth(R_wc, pos, boxes, rng=rng)
        drift = math.radians(imu_yaw_drift_deg_per_s * t)
        Rz = np.array([[math.cos(drift), -math.sin(drift), 0], [math.sin(drift), math.cos(drift), 0], [0, 0, 1]])
        R_imu = Rz @ R_wd
        q = R_to_quat(R_imu if attitude_convention == "device_to_world" else R_imu.T)
        gravity = R_wd.T @ DOWN
        conf = np.where(np.isfinite(depth), 2, 0).astype(np.uint8)
        yield DepthFrame(seq=i, t=t, source="sim", depth=depth, K=K_DEFAULT.copy(), attitude=q, gravity=gravity,
                         confidence=conf, extra={"true_R_wc": R_wc, "true_t": pos})
