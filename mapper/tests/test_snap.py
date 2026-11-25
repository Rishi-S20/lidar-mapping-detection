"""One-click labelling: a click on an object's points gives a box that fits them."""
import numpy as np

from lidar_mapper.detect import build_grid, snap_box


def _room(rng):
    """4 x 4 m floor at z = -1.2, a wall along x = 2, a person (0.5 x 0.4 x 1.7 m) standing against it, a stool."""
    floor = np.column_stack([rng.uniform(-2, 2, (40000, 2)), np.full(40000, -1.2) + rng.normal(0, 0.005, 40000)])
    wall = np.column_stack([rng.uniform(1.9, 2.0, 20000), rng.uniform(-2, 2, 20000), rng.uniform(-1.2, 1.2, 20000)])
    person = np.column_stack([rng.uniform(1.4, 1.9, 8000), rng.uniform(0.0, 0.4, 8000), rng.uniform(-1.2, 0.5, 8000)])
    stool = np.column_stack([rng.uniform(-1.0, -0.6, 3000), rng.uniform(-1.0, -0.6, 3000), rng.uniform(-0.8, -0.75, 3000)])
    return build_grid(np.concatenate([floor, wall, person, stool]))


def test_click_on_a_person_fits_them_and_stops_at_the_wall():
    grid = _room(np.random.default_rng(0))
    box = snap_box(grid, 1.65, 0.2)
    assert abs(box["min"][0] - 1.4) <= 0.1 and box["max"][0] <= 1.95         # does not run into the wall
    assert abs(box["min"][1] - 0.0) <= 0.1 and abs(box["max"][1] - 0.4) <= 0.1
    assert abs(box["height_m"] - 1.7) < 0.1 and abs(box["min"][2] + 1.2) < 0.05


def test_click_on_empty_floor_gives_nothing():
    grid = _room(np.random.default_rng(0))
    assert snap_box(grid, 0.0, 1.5) is None
    stool = snap_box(grid, -0.8, -0.8)
    assert stool["max"][0] - stool["min"][0] < 0.6 and stool["height_m"] < 0.5   # the stool alone, not the room
