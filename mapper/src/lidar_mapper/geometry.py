"""Frames and rotations. World frame: z up (gravity), x/y horizontal, meters (ADR-0015).

Camera frame (OpenCV): x right, y down, z forward, in the depth map's native
landscape orientation. Device frame (CoreMotion): x right, y top of phone,
z out of the screen.

CAM_TO_DEVICE assumes the rear sensor's native orientation is landscape with
the top of the phone on the image's left (the same convention ARKit documents
for `capturedImage`). Verify on a real phone: the `serve` status line's gravity
error must stay small, and the final map report's floor tilt must be < 10 deg.
"""

from __future__ import annotations

import math

import numpy as np

# p_device = CAM_TO_DEVICE @ p_cam   (image right -> device -y, image down -> device -x, forward -> device -z)
CAM_TO_DEVICE = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

DOWN = np.array([0.0, 0.0, -1.0])


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def angle_between(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return math.degrees(math.acos(float(np.clip(c, -1, 1))))


def rotation_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def world_from_device(attitude, gravity) -> tuple[np.ndarray, float]:
    """Device->world rotation from the CoreMotion attitude, disambiguated by gravity.

    CoreMotion documentation is ambiguous about whether its attitude maps
    reference->device or device->reference. We take whichever of R / R^T sends the
    measured gravity (device frame) to world down, and return its error in degrees.
    """
    R = quat_to_R(attitude)
    g = np.asarray(gravity, float)
    if np.linalg.norm(g) < 1e-6:
        return R, float("nan")
    e1, e2 = angle_between(R @ g, DOWN), angle_between(R.T @ g, DOWN)
    return (R, e1) if e1 <= e2 else (R.T, e2)


def backproject(depth: np.ndarray, K: np.ndarray, confidence: np.ndarray | None = None, min_conf: int = 1,
                min_range: float = 0.15, max_range: float = 4.5, stride: int = 1) -> np.ndarray:
    """Depth image -> Nx3 points in the camera (OpenCV) frame."""
    d = depth[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    ok = np.isfinite(d) & (d >= min_range) & (d <= max_range)
    if confidence is not None:
        ok &= confidence[::stride, ::stride] >= min_conf
    z = d[ok]
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = (v[ok] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def exp_so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def log_so3(R: np.ndarray) -> np.ndarray:
    th = math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * th / (2 * math.sin(th))


# ARKit world: y up, x right, -z initial forward. Ours: z up.
ARKIT_TO_WORLD = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
ARKIT_CAM_TO_CV = np.diag([1.0, -1.0, -1.0])


def arkit_pose_to_world(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ARKit camera transform -> (R_world_cam(OpenCV), t). Reference only (camera-based)."""
    R = ARKIT_TO_WORLD @ T[:3, :3] @ ARKIT_CAM_TO_CV
    t = ARKIT_TO_WORLD @ T[:3, 3]
    return R, t


def remove_edge_pixels(depth: np.ndarray, rel_jump: float = 0.05) -> np.ndarray:
    """Drop "flying pixels": depth readings at object edges that blend foreground and background.

    A pixel is dropped if any 4-neighbour differs from it by more than `rel_jump` x its depth.
    These land in mid-air between objects and are the main source of scattered map points.
    """
    d = depth.copy()
    bad = np.zeros(d.shape, bool)
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(depth, shift, axis)
            with np.errstate(invalid="ignore"):
                bad |= np.abs(depth - nb) > rel_jump * depth
    bad[0, :] = bad[-1, :] = bad[:, 0] = bad[:, -1] = True  # roll wraps around at the borders
    d[bad] = np.nan
    return d


def remove_grazing_pixels(depth: np.ndarray, K: np.ndarray, min_cos: float = 0.25) -> np.ndarray:
    """Drop pixels whose surface is seen nearly edge-on (angle to the ray > ~75 deg at min_cos=0.25).

    Depth on grazing surfaces is very noisy and smears along the ray; those readings make the
    wispy "sheets" floating in free space. Normals come from neighbouring pixels.
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], -1)
    du = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    dv = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    n = np.cross(du, dv)
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        ray = P / np.linalg.norm(P, axis=-1, keepdims=True)
        cos = np.abs(np.sum(n * ray, -1))
    d = depth.copy()
    d[~(cos >= min_cos)] = np.nan  # also drops pixels whose normal is undefined (holes, borders)
    return d
