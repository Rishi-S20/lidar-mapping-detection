"""Person detector v0 on the LiDAR map: geometry only, so it works in the dark.

Input is a finished scan: the voxel map in `<scan>.live.npz` and, for training,
the boxes in `<scan>.labels.json`.

How it works:
  1. Floor height is found from the map. Points 7 cm to 2 m above it are kept,
     so the ceiling never matters.
  2. The floor plan is cut into 10 cm cells. Each cell gets a height profile
     (how many map voxels in 8 height bands), its top height, and the same
     numbers for the area around it (0.3 to 0.65 m).
  3. A random forest scores every occupied cell: person or not. Training cells
     inside a `person` box are positives, everything else is a negative.
  4. Neighbouring high-score cells are grouped into detections with a box.

Usage:
  python -m lidar_mapper.detect eval  recordings recordings/synthetic
  python -m lidar_mapper.detect train recordings recordings/synthetic --model models/person-v0.joblib
  python -m lidar_mapper.detect run   recordings/<scan>.live.npz --model models/person-v0.joblib

Needs scikit-learn (pip install -e ".[detect]").
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

CELL = 0.10                      # m, floor-plan cell
MIN_H, MAX_H = 0.07, 2.0         # m above the floor; below is floor noise, above is ceiling
BANDS = np.array([MIN_H, 0.2, 0.35, 0.5, 0.75, 1.0, 1.3, 1.6, MAX_H])
WALL_H = 1.85                    # a cell reaching this height is wall or tall furniture
MIN_CELL_VOXELS = 3              # occupied cells need this many map voxels
MIN_DET_CELLS = 8                # 0.08 m2: smaller blobs are dropped
FEATURES = (["top", "mean_h"] + [f"band{i}" for i in range(8)] + [f"near_band{i}" for i in range(8)]
            + ["top_mean3", "top_mean7", "top_mean13", "top_max7", "top_max13", "occ7", "occ13",
               "wall7", "wall13", "top_std7", "low7", "low13"])


def floor_height(points: np.ndarray) -> float:
    """Densest 2 cm layer among the lowest 30% of the map."""
    z = points[:, 2]
    low = z[z < np.quantile(z, 0.3)]
    hist, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.02, 0.02))
    return float(edges[hist.argmax()] + 0.01)


@dataclass
class Grid:
    """One scan as floor-plan cells with features."""
    origin: np.ndarray           # world x, y of cell (0, 0)
    floor: float
    occupied: np.ndarray         # (nx, ny) bool
    top: np.ndarray              # (nx, ny) top height above floor, 0 where empty
    features: np.ndarray         # (n_occupied, len(FEATURES))
    cells: np.ndarray            # (n_occupied, 2) cell indices

    def centres(self) -> np.ndarray:
        return self.origin + (self.cells + 0.5) * CELL


def build_grid(points: np.ndarray, hits: np.ndarray | None = None) -> Grid:
    if hits is not None:
        points = points[hits >= 2]       # single-hit voxels are mostly noise
    floor = floor_height(points)
    h = points[:, 2] - floor
    p, h = points[(h > MIN_H) & (h < MAX_H)], h[(h > MIN_H) & (h < MAX_H)]
    origin = np.floor(p[:, :2].min(0) / CELL) * CELL
    ij = np.floor((p[:, :2] - origin) / CELL).astype(int)
    shape = tuple(ij.max(0) + 1)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    size = shape[0] * shape[1]
    band = np.clip(np.searchsorted(BANDS, h, side="right") - 1, 0, 7)
    counts = np.stack([np.bincount(flat[band == b], minlength=size) for b in range(8)], -1).reshape(*shape, 8)
    total = counts.sum(-1)
    occupied = total >= MIN_CELL_VOXELS
    top = np.zeros(size)
    np.maximum.at(top, flat, h)
    top = np.where(occupied, top.reshape(shape), 0.0)
    mean_h = np.where(occupied, np.bincount(flat, weights=h, minlength=size).reshape(shape) / np.maximum(total, 1), 0.0)

    def mean(a, n):
        return ndimage.uniform_filter(a.astype(float), n, mode="constant")

    counts = np.where(occupied[..., None], counts, 0)
    near = np.stack([mean(counts[..., b], 5) * 25 for b in range(8)], -1)
    wall = (top > WALL_H).astype(float)
    low = (occupied & (top < 0.5)).astype(float)
    maps = [top, mean_h] + [np.log1p(counts[..., b]) for b in range(8)] + [np.log1p(near[..., b]) for b in range(8)]
    maps += [mean(top, 3), mean(top, 7), mean(top, 13),
             ndimage.maximum_filter(top, 7, mode="constant"), ndimage.maximum_filter(top, 13, mode="constant"),
             mean(occupied, 7), mean(occupied, 13), mean(wall, 7), mean(wall, 13),
             np.sqrt(np.maximum(mean(top ** 2, 7) - mean(top, 7) ** 2, 0)), mean(low, 7), mean(low, 13)]
    cells = np.argwhere(occupied)
    features = np.stack([m[occupied] for m in maps], -1).astype(np.float32)
    return Grid(origin, floor, occupied, top, features, cells)


def cell_labels(grid: Grid, boxes: list[dict], label: str = "person") -> np.ndarray:
    """1 for occupied cells whose centre is inside a box of this label and that reach into its height range."""
    y = np.zeros(len(grid.cells), bool)
    c = grid.centres()
    tops = grid.top[grid.cells[:, 0], grid.cells[:, 1]]
    for b in boxes:
        if b.get("label") != label:
            continue
        lo, hi = np.asarray(b["min"], float), np.asarray(b["max"], float)
        y |= np.all((c >= lo[:2]) & (c <= hi[:2]), 1) & (tops + grid.floor >= lo[2])
    return y


@dataclass
class Scan:
    name: str
    grid: Grid
    y: np.ndarray
    boxes: list[dict]            # person boxes only
    origin: str                  # "real" or "synthetic"
    room: str
    base: str                    # for synthetic scans, the real scan they were built on


def load_scans(paths: list[str], skip: tuple[str, ...] = ()) -> list[Scan]:
    """Every labelled scan under these folders or files. Scans without a labels file are skipped."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        files += sorted(path.glob("*.live.npz")) if path.is_dir() else [path]
    scans = []
    for f in files:
        name = f.name[: -len(".live.npz")]
        label_file = f.with_name(name + ".labels.json")
        if name in skip or not label_file.exists():
            continue
        labels = json.loads(label_file.read_text())
        scene = labels.get("scene", {})
        if not scene.get("scene"):
            continue                     # never labelled: we can't call it a negative
        data = np.load(f)
        grid = build_grid(data["map_points"], data["map_hits"])
        boxes = [b for b in labels.get("boxes", []) if b.get("label") == "person"]
        scans.append(Scan(name, grid, cell_labels(grid, boxes), boxes, scene.get("data_origin", "real"),
                          scene.get("room", ""), Path(scene.get("base_scan", "")).stem))
    return scans


def fit(scans: list[Scan], seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    X = np.concatenate([s.grid.features for s in scans])
    y = np.concatenate([s.y for s in scans])
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=4, max_features="sqrt",
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    return model.fit(X, y)


def detect(grid: Grid, model, threshold: float = 0.5) -> list[dict]:
    """Group high-score cells into detections: box in world metres, score, footprint in cells."""
    if len(grid.cells) == 0:
        return []
    prob = np.zeros(grid.occupied.shape)
    prob[grid.cells[:, 0], grid.cells[:, 1]] = model.predict_proba(grid.features)[:, 1]
    smooth = ndimage.uniform_filter(prob, 3, mode="constant")
    blobs, n = ndimage.label((smooth >= threshold) & grid.occupied, structure=np.ones((3, 3)))
    found = []
    for i in range(1, n + 1):
        cells = np.argwhere(blobs == i)
        if len(cells) < MIN_DET_CELLS:
            continue
        lo = grid.origin + cells.min(0) * CELL
        hi = grid.origin + (cells.max(0) + 1) * CELL
        height = float(grid.top[cells[:, 0], cells[:, 1]].max())
        found.append({"label": "person", "score": round(float(prob[cells[:, 0], cells[:, 1]].mean()), 3),
                      "min": [round(float(lo[0]), 3), round(float(lo[1]), 3), round(grid.floor, 3)],
                      "max": [round(float(hi[0]), 3), round(float(hi[1]), 3), round(grid.floor + height, 3)],
                      "cells": len(cells), "height_m": round(height, 2)})
    return sorted(found, key=lambda d: -d["score"])


def person_range(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, boxes: list[dict],
                 half_angle_deg: float = 10.0, min_share: float = 0.15, front_margin: float = 0.15) -> dict:
    """Distance from the phone to a detected person it is pointing at, from one depth frame.

    `boxes` are detections in world metres; R, t are world <- camera. Only depth pixels within
    `half_angle_deg` of the camera axis are used.

      no_person   no detection box is in that cone
      ambiguous   a person is in the cone but fills too little of it, or something is nearer
      person      the person is the nearest thing in the cone: `range_m` is the distance to them
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    cone = ((np.arccos(1 / np.sqrt(x * x + y * y + 1)) <= np.radians(half_angle_deg))
            & np.isfinite(depth) & (depth > 0.05))
    out = {"state": "no_person", "range_m": None, "nearest_m": None, "cone_share": 0.0}
    if cone.sum() < 20 or not boxes:
        return out
    d = depth[cone]
    cam = np.stack([x[cone] * d, y[cone] * d, d], -1)
    ranges = np.linalg.norm(cam, axis=1)
    world = cam @ np.asarray(R).T + np.asarray(t)
    nearest = float(np.percentile(ranges, 10))
    out["nearest_m"] = round(nearest, 3)
    best = None
    for box in boxes:
        inside = np.all((world >= np.asarray(box["min"]) - 0.05) & (world <= np.asarray(box["max"]) + 0.05), axis=1)
        if inside.sum() >= 20 and (best is None or inside.mean() > best[0]):
            best = (float(inside.mean()), float(np.percentile(ranges[inside], 10)))
    if best is None:
        return out
    share, person_near = best
    out["cone_share"] = round(share, 2)
    if share < min_share or person_near > nearest + front_margin:
        out["state"] = "ambiguous"
    else:
        out.update(state="person", range_m=round(person_near, 3))
    return out


def _overlap(det: dict, box: dict) -> float:
    """Share of the detection's footprint that lies inside the labelled box."""
    lo = np.maximum(det["min"][:2], np.asarray(box["min"][:2]))
    hi = np.minimum(det["max"][:2], np.asarray(box["max"][:2]))
    area = np.prod(np.asarray(det["max"][:2]) - np.asarray(det["min"][:2]))
    return float(np.prod(np.clip(hi - lo, 0, None)) / max(area, 1e-9))


def score_scan(scan: Scan, detections: list[dict]) -> dict:
    """A box is found when a detection has half its footprint inside it; other detections are false alarms."""
    hit = [any(_overlap(d, b) >= 0.5 for d in detections) for b in scan.boxes]
    false = [d for d in detections if not any(_overlap(d, b) >= 0.5 for b in scan.boxes)]
    return {"boxes": len(scan.boxes), "found": int(sum(hit)), "false_alarms": len(false)}


def evaluate(scans: list[Scan], by: str = "scan", use_synthetic: bool = True, threshold: float = 0.5) -> list[dict]:
    """Hold out real scans in turn (one scan, or one whole room), train on the rest, test on the held-out ones."""
    real = [s for s in scans if s.origin == "real"]
    synth = [s for s in scans if s.origin != "real"] if use_synthetic else []
    groups = sorted({s.name if by == "scan" else s.room for s in real})
    rows = []
    for g in groups:
        test = [s for s in real if (s.name if by == "scan" else s.room) == g]
        held = {s.name for s in test}
        # A synthetic scan built on a held-out scan shows the same room, so it is held out too.
        train = [s for s in real if s.name not in held] + [s for s in synth if s.base not in held]
        if not any(s.y.any() for s in train):
            continue
        model = fit(train)
        for s in test:
            rows.append({"scan": s.name, "room": s.room, **score_scan(s, detect(s.grid, model, threshold))})
    return rows


def summarize(rows: list[dict]) -> dict:
    boxes = sum(r["boxes"] for r in rows)
    return {"scans": len(rows), "people": boxes, "found": sum(r["found"] for r in rows),
            "false_alarms": sum(r["false_alarms"] for r in rows),
            "scans_with_false_alarm": sum(r["false_alarms"] > 0 for r in rows)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lidar_mapper.detect", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "train"):
        p = sub.add_parser(name)
        p.add_argument("paths", nargs="+", help="folders with <scan>.live.npz and <scan>.labels.json")
        p.add_argument("--skip", default="", help="comma-separated scan names to leave out")
        p.add_argument("--threshold", type=float, default=0.5)
    sub.choices["train"].add_argument("--model", required=True)
    sub.choices["eval"].add_argument("--json", help="write the per-scan rows here")
    p = sub.add_parser("run")
    p.add_argument("scan", help="<scan>.live.npz")
    p.add_argument("--model", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        import joblib
        data = np.load(a.scan)
        print(json.dumps(detect(build_grid(data["map_points"], data["map_hits"]), joblib.load(a.model), a.threshold),
                         indent=1))
        return
    scans = load_scans(a.paths, tuple(x for x in a.skip.split(",") if x))
    real = [s for s in scans if s.origin == "real"]
    print(f"{len(real)} real scans ({sum(len(s.boxes) for s in real)} people), "
          f"{len(scans) - len(real)} synthetic ({sum(len(s.boxes) for s in scans if s.origin != 'real')} people)")
    if a.cmd == "train":
        import joblib
        Path(a.model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit(scans), a.model, compress=3)
        print(f"saved {a.model}")
        return
    out = {}
    for by in ("scan", "room"):
        for use_synth in (True, False):
            rows = evaluate(scans, by, use_synth, a.threshold)
            key = f"hold out one {by}, {'real + synthetic' if use_synth else 'real only'} training"
            out[key] = rows
            print(f"\n{key}: {summarize(rows)}")
            for r in rows:
                print(f"  {r['scan']:28s} {r['room']:8s} people {r['found']}/{r['boxes']}  false alarms {r['false_alarms']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
