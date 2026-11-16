import numpy as np

from lidar_mapper import protocol
from lidar_mapper.geometry import (CAM_TO_DEVICE, R_to_quat, backproject, quat_to_R, rotation_angle_deg,
                                   world_from_device)
from lidar_mapper.mapper import Mapper, floor_check
from lidar_mapper.sim import K_DEFAULT, default_room, device_rotation, render_depth, stream


def test_cam_to_device_is_a_rotation():
    assert np.allclose(CAM_TO_DEVICE @ CAM_TO_DEVICE.T, np.eye(3))
    assert np.isclose(np.linalg.det(CAM_TO_DEVICE), 1)


def test_quaternion_roundtrip():
    R = device_rotation(37, 12)
    assert np.allclose(quat_to_R(R_to_quat(R)), R, atol=1e-9)


def test_world_from_device_handles_both_conventions():
    R = device_rotation(20, 30)
    g = R.T @ np.array([0, 0, -1.0])
    for q in (R_to_quat(R), R_to_quat(R.T)):
        Rw, err = world_from_device(q, g)
        assert err < 1e-6 and np.allclose(Rw, R, atol=1e-9)


def test_protocol_roundtrip():
    f = next(stream(1))
    g = protocol.decode(protocol.encode(f))
    assert g.depth.shape == (192, 256) and g.confidence is not None
    assert np.allclose(np.nan_to_num(g.depth), np.nan_to_num(f.depth), atol=2e-3)  # float16 at <5 m
    assert np.allclose(g.attitude, f.attitude) and np.allclose(g.K, f.K)


def test_backprojection_puts_floor_below():
    R_wc = device_rotation(0, 30) @ CAM_TO_DEVICE
    depth = render_depth(R_wc, np.zeros(3), default_room(), noise=0, dropout=0)
    pts_w = backproject(depth, K_DEFAULT) @ R_wc.T
    assert abs(np.percentile(pts_w[:, 2], 2) - (-1.3)) < 0.03  # floor at z = -1.3


def test_mapper_tracks_a_walk_in_the_dark():
    m = Mapper()
    errs, statuses = [], []
    for f in stream(50):
        r = m.process(f)
        statuses.append(r.status)
        errs.append(np.linalg.norm(r.t - f.extra["true_t"]))
        rot = rotation_angle_deg(r.R @ f.extra["true_R_wc"].T)
        assert rot < 4, (f.seq, rot)
    assert statuses.count("tracking") >= 45
    assert max(errs) < 0.12, max(errs)  # walked 1.2 m; IMU yaw drifts 0.5 deg/s
    pts, _ = m.map.points(2)
    fc = floor_check(pts)
    assert fc["ok"] and abs(fc["floor_z"] + 1.3) < 0.05, fc


def test_mapper_with_reference_frame_convention():
    m = Mapper()
    last = None
    for f in stream(30, attitude_convention="world_to_device"):
        last = m.process(f)
    assert np.linalg.norm(last.t - f.extra["true_t"]) < 0.12


def test_lost_when_scene_is_empty():
    m = Mapper()
    f = next(stream(1))
    m.process(f)
    f2 = next(stream(2, seed=1))
    f2.depth[:] = np.nan
    f2.t = 0.1
    r = m.process(f2)
    assert r.status == "lost"


def test_waits_for_real_depth_before_starting_the_map():
    m = Mapper()
    frames = list(stream(3))
    empty = frames[0]
    empty.depth[:] = np.nan
    assert m.process(empty).status == "waiting" and len(m.map) == 0
    assert m.process(frames[1]).status == "init" and len(m.map) > 0


def _restarted_session(frames, start, yaw_jump_deg):
    """Frames from `start` on, as if the app restarted: seq from 0, a time gap, IMU yaw jumped."""
    import math
    from lidar_mapper.geometry import R_to_quat, quat_to_R
    y = math.radians(yaw_jump_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    out = []
    for i, f in enumerate(frames[start:]):
        f.seq, f.t = i, f.t + 30.0
        f.attitude = R_to_quat(Rz @ quat_to_R(f.attitude))
        out.append(f)
    return out


def test_restart_relocalizes_in_a_mapped_area():
    frames = list(stream(60))
    m = Mapper()
    for f in frames[:40]:
        m.process(f)
    statuses = [m.process(f).status for f in _restarted_session(frames, 30, 70)]
    assert "relocalized" in statuses
    assert statuses[-1] == "tracking" and not m.segments


def test_new_segment_then_merge():
    frames = list(stream(80))
    m = Mapper()
    m.min_merge_voxels = 5000
    for f in frames[:25]:
        m.process(f)
    sess2 = _restarted_session(frames, 45, 70)
    m._relocalize = lambda pts, R, yaw_known=False: None  # force: relocalization fails -> new segment
    statuses = [m.process(f).status for f in sess2]
    assert "merged" in statuses, statuses
    assert not m.segments


def test_export_writes_parquet_and_manifest(tmp_path):
    import json

    import pyarrow.parquet as pq

    from lidar_mapper.export import export_session
    from lidar_mapper.server import write_record

    esd = tmp_path / "simscan.esd"
    with open(esd, "wb") as fh:
        for f in stream(25):
            write_record(fh, protocol.encode(f))
    res = export_session(esd, tmp_path / "out", labels={"lights": "off"})
    fr = pq.read_table(res.frames_path).to_pydict()
    assert len(fr["seq"]) == 25 and fr["session_id"][0] == res.session_id
    d = np.frombuffer(fr["depth_f16"][3], "<f2").reshape(fr["height"][3], fr["width"][3])
    assert d.shape == (192, 256) and (d > 0).mean() > 0.5
    assert len(fr["pose_R"][3]) == 9 and fr["pose_status"].count("tracking") >= 20
    mp = pq.read_table(res.map_path).to_pydict()
    assert len(mp["x"]) > 1000 and set(mp["segment"]) == {0}
    meta = json.loads(res.session_path.read_text())
    assert meta["labels"] == {"lights": "off"} and meta["stats"]["frames"] == 25


def test_clear_map_starts_fresh_recording(tmp_path):
    from lidar_mapper.server import LiveMapper
    live = LiveMapper(Mapper(), record=str(tmp_path / "rec.esd"), verbose=False)
    for f in stream(10):
        live.ingest(protocol.encode(f))
    assert len(live.mapper.map) > 0
    live.clear()
    assert len(live.mapper.map) == 0 and live.mapper.R is None
    assert (tmp_path / "rec-scan2.esd").exists()
    for f in stream(10):  # a new scan maps from scratch (seq restarts, no relocalization needed)
        live.ingest(protocol.encode(f))
    assert len(live.mapper.map) > 0 and not live.mapper.segments
    live.close()


def test_labels_saved_live_and_exported(tmp_path):
    import json

    import pyarrow.parquet as pq

    from lidar_mapper.export import export_session
    from lidar_mapper.server import LiveMapper

    rec = tmp_path / "labelled.esd"
    live = LiveMapper(Mapper(), record=str(rec), verbose=False)
    from lidar_mapper.server import write_record
    for f in stream(20):
        msg = protocol.encode(f)
        write_record(live.record_fh, msg)  # what phone_handler does
        live.ingest(msg)
    live.record_fh.flush()
    saved = live.set_labels({"lights": "off", "scene": "person_lying"},
                            [{"label": "person", "pose": "lying", "min": [1.2, -0.4, -1.3], "max": [2.0, 0.2, -1.05]},
                             {"label": "bad", "min": [1, 1, 1], "max": [0, 0, 0]}])  # invalid box is dropped
    assert any(s.endswith(".labels.json") for s in saved) and any(s.endswith(".live.npz") for s in saved)
    live.close()
    res = export_session(rec, tmp_path / "out", labels={"room": "sim"})
    meta = json.loads(res.session_path.read_text())
    assert meta["map_source"] == "live"
    assert meta["labels"] == {"lights": "off", "scene": "person_lying", "room": "sim"}
    assert len(meta["annotations"]) == 1 and meta["annotations"][0]["label"] == "person"
    fr = pq.read_table(res.frames_path).to_pydict()
    assert fr["pose_status"].count("tracking") >= 15
    mp = pq.read_table(res.map_path).to_pydict()
    assert len(mp["x"]) == len(np.load(rec.with_suffix(".live.npz"))["map_points"])
