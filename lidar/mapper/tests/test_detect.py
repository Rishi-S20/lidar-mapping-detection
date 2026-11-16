"""Person detector v0: the pipeline runs end to end on a made-up room. Says nothing about real accuracy."""
import numpy as np
import pytest

pytest.importorskip("sklearn")
from lidar_mapper.detect import Scan, build_grid, cell_labels, detect, fit, score_scan  # noqa: E402


def _room(rng, mound_at):
    """4 x 4 m floor at z = -1.2 with a low table, plus a body-sized mound (1.6 x 0.5 x 0.3 m) if asked."""
    floor = np.column_stack([rng.uniform(-2, 2, (40000, 2)), np.full(40000, -1.2) + rng.normal(0, 0.005, 40000)])
    table = np.column_stack([rng.uniform(1.0, 1.8, 6000), rng.uniform(1.0, 1.8, 6000), rng.uniform(-0.5, -0.45, 6000)])
    parts, boxes = [floor, table], []
    if mound_at is not None:
        x, y = mound_at
        u, v = rng.uniform(-0.8, 0.8, 8000), rng.uniform(-0.25, 0.25, 8000)
        top = -1.2 + 0.3 * np.sqrt(np.clip(1 - (u / 0.8) ** 2 - (v / 0.25) ** 2, 0, 1))
        parts.append(np.column_stack([x + u, y + v, rng.uniform(-1.2, 1, 8000) * 0 + rng.uniform(-1.2, top)]))
        boxes = [{"label": "person", "min": [x - 0.85, y - 0.3, -1.2], "max": [x + 0.85, y + 0.3, -0.85]}]
    grid = build_grid(np.concatenate(parts))
    return Scan("s", grid, cell_labels(grid, boxes), boxes, "real", "r", "")


def test_detector_finds_a_mound_it_was_trained_on_elsewhere():
    rng = np.random.default_rng(0)
    train = [_room(rng, (-1, -1)), _room(rng, (0.5, -1.2)), _room(rng, None), _room(rng, (-0.8, 0.9))]
    model = fit(train)
    with_person, empty = _room(rng, (0, -0.2)), _room(rng, None)
    assert score_scan(with_person, detect(with_person.grid, model))["found"] == 1
    assert detect(empty.grid, model) == []


def _wall_frame(person_at=None, blocker_at=None):
    """Depth frame looking down +z at a wall 3 m away, optionally with a person-sized slab and a nearer blocker."""
    K = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])
    depth = np.full((240, 320), 3.0)
    if person_at:
        depth[60:200, 120:200] = person_at
    if blocker_at:
        depth[100:140, 150:170] = blocker_at
    return depth, K


def test_distance_to_a_person_only_when_they_are_the_nearest_thing_ahead():
    from lidar_mapper.detect import person_range
    R, t = np.eye(3), np.zeros(3)
    box = [{"min": [-0.3, -0.5, 1.6], "max": [0.3, 0.6, 2.0]}]
    depth, K = _wall_frame(person_at=1.8)
    hit = person_range(depth, K, R, t, box)
    assert hit["state"] == "person" and abs(hit["range_m"] - 1.8) < 0.05
    blocked, _ = _wall_frame(person_at=1.8, blocker_at=0.9)
    assert person_range(blocked, K, R, t, box)["state"] == "ambiguous"        # something nearer is in the way
    away = [{"min": [2.0, -0.5, 1.6], "max": [2.6, 0.6, 2.0]}]
    assert person_range(depth, K, R, t, away)["state"] == "no_person"         # not aimed at the box
