"""Depth stream v0.1 (see PROTOCOL.md): parse and build binary frame messages."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np

VERSION = "0.1"


@dataclass
class DepthFrame:
    seq: int
    t: float
    source: str
    depth: np.ndarray  # float32 [h, w], meters, nan = no return
    K: np.ndarray  # 3x3 intrinsics in depth-map pixels
    attitude: np.ndarray  # quaternion [w, x, y, z]
    gravity: np.ndarray  # device frame, g
    confidence: np.ndarray | None = None  # uint8 [h, w]
    rotation_rate: np.ndarray | None = None
    user_accel: np.ndarray | None = None
    arkit_pose: np.ndarray | None = None  # 4x4, reference only
    arkit_tracking: str | None = None
    ambient_lux: float | None = None
    filtered: bool = False
    attitude_ref: str = "arbitrary"  # "magnetic_north": yaw is consistent across app restarts
    extra: dict = field(default_factory=dict)

    @property
    def valid_fraction(self) -> float:
        return float(np.isfinite(self.depth).mean()) if self.depth.size else 0.0


def _vec(v, n):
    if v is None:
        return None
    a = np.asarray(v, float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"expected {n} finite numbers, got {v!r}")
    return a


def decode(message: bytes) -> DepthFrame:
    if len(message) < 4:
        raise ValueError("message too short")
    (hlen,) = struct.unpack_from("<I", message, 0)
    if hlen <= 0 or 4 + hlen > len(message):
        raise ValueError("bad header length")
    hdr = json.loads(message[4:4 + hlen].decode("utf-8"))
    if hdr.get("v") != VERSION:
        raise ValueError(f"unsupported depth stream version {hdr.get('v')!r}")
    w, h = int(hdr["w"]), int(hdr["h"])
    if not (0 < w <= 4096 and 0 < h <= 4096):
        raise ValueError("bad depth size")
    off = 4 + hlen
    n = w * h
    expect = off + 2 * n + (n if hdr.get("confidence") else 0)
    if len(message) != expect:
        raise ValueError(f"payload size {len(message)} != expected {expect}")
    depth = np.frombuffer(message, "<f2", n, off).astype(np.float32).reshape(h, w)
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    conf = None
    if hdr.get("confidence"):
        conf = np.frombuffer(message, np.uint8, n, off + 2 * n).reshape(h, w).copy()
    K = np.array([[hdr["fx"], 0, hdr["cx"]], [0, hdr["fy"], hdr["cy"]], [0, 0, 1]], float)
    pose = hdr.get("arkit_pose")
    return DepthFrame(
        seq=int(hdr["seq"]), t=float(hdr["t"]), source=str(hdr["source"]), depth=depth, K=K,
        attitude=_vec(hdr["attitude"], 4), gravity=_vec(hdr["gravity"], 3), confidence=conf,
        rotation_rate=_vec(hdr.get("rotation_rate"), 3), user_accel=_vec(hdr.get("user_accel"), 3),
        arkit_pose=np.asarray(pose, float).reshape(4, 4, order="F") if pose is not None else None,
        arkit_tracking=hdr.get("arkit_tracking"), ambient_lux=hdr.get("ambient_lux"),
        filtered=bool(hdr.get("filtered", False)), attitude_ref=str(hdr.get("attitude_ref", "arbitrary")),
    )


def encode(f: DepthFrame) -> bytes:
    """Inverse of decode (used by tests, the simulator, and replay)."""
    h, w = f.depth.shape
    hdr = {
        "v": VERSION, "seq": f.seq, "t": f.t, "source": f.source, "w": w, "h": h,
        "fx": float(f.K[0, 0]), "fy": float(f.K[1, 1]), "cx": float(f.K[0, 2]), "cy": float(f.K[1, 2]),
        "confidence": f.confidence is not None, "filtered": f.filtered,
        "attitude": [float(x) for x in f.attitude], "gravity": [float(x) for x in f.gravity],
        "rotation_rate": None if f.rotation_rate is None else [float(x) for x in f.rotation_rate],
        "user_accel": None if f.user_accel is None else [float(x) for x in f.user_accel],
        "arkit_pose": None if f.arkit_pose is None else [float(x) for x in f.arkit_pose.reshape(-1, order="F")],
        "arkit_tracking": f.arkit_tracking, "ambient_lux": f.ambient_lux, "attitude_ref": f.attitude_ref,
    }
    hb = json.dumps(hdr).encode("utf-8")
    d = np.nan_to_num(f.depth, nan=0.0).astype("<f2").tobytes()
    c = f.confidence.astype(np.uint8).tobytes() if f.confidence is not None else b""
    return struct.pack("<I", len(hb)) + hb + d + c
