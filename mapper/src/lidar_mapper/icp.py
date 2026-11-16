"""Point-to-plane ICP (numpy + scipy). Aligns a new depth frame to the map built so far.

This is the dark-safe position estimate: it uses only LiDAR geometry, with the
IMU rotation as the starting guess and a soft prior (ADR-0015).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import exp_so3, log_so3


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0:
        return points
    k = np.floor(points / voxel).astype(np.int64) + (1 << 20)
    keys = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]  # pack: 1-D unique is ~10x faster than axis=0
    _, idx = np.unique(keys, return_index=True)
    return points[idx]


def estimate_normals(points: np.ndarray, tree: cKDTree, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals and planarity (0..1) by PCA of the k nearest neighbours."""
    k = min(k, len(points))
    _, nn = tree.query(points, k=k, workers=-1)
    nb = points[nn]  # [n, k, 3]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    evals, evecs = np.linalg.eigh(cov)  # ascending
    normals = evecs[:, :, 0]
    planarity = 1 - evals[:, 0] / np.maximum(evals[:, 1], 1e-12)
    return normals, planarity


@dataclass
class ICPResult:
    R: np.ndarray
    t: np.ndarray
    rmse: float
    inlier_frac: float
    iterations: int
    converged: bool


def icp_point_to_plane(src: np.ndarray, tgt: np.ndarray, tgt_normals: np.ndarray, tree: cKDTree,
                       R0: np.ndarray, t0: np.ndarray, max_iter: int = 25, max_dist: float = 0.2,
                       min_dist: float = 0.05, rot_prior: np.ndarray | None = None,
                       rot_prior_weight: float = 0.0, trans_damping: float = 1e-3,
                       tilt_prior_weight: float = 0.0) -> ICPResult:
    """Find R, t minimizing sum((n_q . (R p + t - q))^2) over matched pairs.

    rot_prior: rotation the result should stay close to (e.g. IMU), weighted by
    rot_prior_weight (in residual units per radian, scaled by the number of matches).
    tilt_prior_weight: extra weight on the roll/pitch part of that prior (rotation about the
    world x/y axes). Gravity measures tilt to ~0.5 deg, far better than ICP on a few
    surfaces, so tilt is held to the IMU while yaw and position stay free.
    trans_damping keeps translation stable along directions the scene doesn't constrain
    (a single flat wall).
    """
    R, t = R0.copy(), t0.astype(float).copy()
    dist = max_dist
    rmse, inl, it, converged = float("inf"), 0.0, 0, False
    for it in range(1, max_iter + 1):
        p = src @ R.T + t
        d, j = tree.query(p, distance_upper_bound=dist)
        m = np.isfinite(d)
        inl = float(m.mean()) if len(m) else 0.0
        if m.sum() < 30:
            break
        pm, q, n = p[m], tgt[j[m]], tgt_normals[j[m]]
        r = np.einsum("ij,ij->i", n, pm - q)
        # Huber weights
        s = 0.02
        wts = np.where(np.abs(r) <= s, 1.0, s / np.maximum(np.abs(r), 1e-12))
        J = np.hstack([np.cross(pm, n), n])  # d r / d [omega, tau]
        A = (J * wts[:, None]).T @ J
        b = -(J * wts[:, None]).T @ r
        A[3:, 3:] += np.eye(3) * trans_damping * len(r)
        if rot_prior is not None and (rot_prior_weight > 0 or tilt_prior_weight > 0):
            lam = np.array([rot_prior_weight + tilt_prior_weight] * 2 + [rot_prior_weight]) * len(r)
            e = log_so3(rot_prior @ R.T)  # rotation (world frame) still needed to reach the prior
            A[:3, :3] += np.diag(lam)
            b[:3] += lam * e
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        dR = exp_so3(x[:3])
        R = dR @ R
        t = dR @ t + x[3:]
        rmse = float(np.sqrt(np.mean(r ** 2)))
        dist = max(min_dist, dist * 0.7)
        if np.linalg.norm(x[:3]) < 1e-4 and np.linalg.norm(x[3:]) < 1e-4:
            converged = True
            break
    # final statistics at the solution
    p = src @ R.T + t
    d, j = tree.query(p, distance_upper_bound=min_dist)
    m = np.isfinite(d)
    if m.any():
        r = np.einsum("ij,ij->i", tgt_normals[j[m]], p[m] - tgt[j[m]])
        rmse = float(np.sqrt(np.mean(r ** 2)))
    inl = float(m.mean()) if len(m) else 0.0
    return ICPResult(R, t, rmse, inl, it, converged)
