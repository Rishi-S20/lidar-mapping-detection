"""Metadata regression tests. Every depth/range fixture here is SYNTHETIC."""
import json
import asyncio
from dataclasses import replace

import pytest

from lidar_mapper import protocol
from lidar_mapper.mapper import Mapper
from lidar_mapper.server import LiveMapper, write_record
from lidar_mapper.sim import stream


@pytest.fixture(scope="module")
def frame():
    return next(stream(1))


@pytest.mark.parametrize("delivery,sources,expected", [
    ("live", ["lidar"], "live"),
    ("live", ["arkit"], "live"),
    ("live", ["sim"], "simulated"),
    ("recorded", ["sim"], "simulated"),
    ("recorded", ["lidar"], "live"),
    ("recorded", ["legacy-unknown"], "recorded"),
    ("live", ["legacy-unknown"], "recorded"),
    ("live", ["sim", "lidar"], "simulated"),
    ("live", ["lidar", "sim"], "simulated"),
    ("live", ["lidar", "legacy-unknown"], "recorded"),
    ("simulated", [], "simulated"),
    ("live", [], "recorded"),
    ("recorded", [], "recorded"),
])
def test_snapshot_preserves_delivery_and_accumulated_acquisition(frame, delivery, sources, expected):
    live = LiveMapper(Mapper(), delivery=delivery)
    for seq, source in enumerate(sources):
        live.ingest(protocol.encode(replace(frame, source=source, seq=seq)))
    raw = json.loads(live.snapshot())
    assert set(raw) == {"protocol_version", "type", "sequence", "timestamp", "source",
                        "acquisition_source", "frame", "detections"}
    assert raw["protocol_version"] == "0.1" and raw["type"] == "sonar_frame"
    assert raw["source"] == delivery
    assert raw["acquisition_source"] == raw["frame"]["meta"]["source"] == expected
    assert all(len(row) == 4 for row in raw["frame"]["points"])


def test_metadata_does_not_change_geometry(frame):
    frames = []
    for delivery in ("live", "simulated", "recorded"):
        live = LiveMapper(Mapper(), delivery=delivery)
        live.ingest(protocol.encode(frame))
        frames.append(json.loads(live.snapshot())["frame"]["points"])
    assert frames[0] == frames[1] == frames[2]


def test_invalid_delivery_is_rejected():
    with pytest.raises(ValueError, match="delivery"):
        LiveMapper(Mapper(), delivery="invented")


@pytest.mark.parametrize("command,source,delivery,acquisition", [
    ("sim", "sim", "simulated", "simulated"),
    ("replay", "sim", "recorded", "simulated"),
    ("replay", "lidar", "recorded", "live"),
    ("replay", "unknown", "recorded", "recorded"),
])
def test_cli_served_modes(monkeypatch, tmp_path, frame, command, source, delivery, acquisition):
    from lidar_mapper import __main__ as cli
    captured = []

    async def paced(live, messages, port):
        live.ingest(messages[0])
        captured.append(json.loads(live.snapshot()))

    monkeypatch.setattr(cli, "_paced", paced)
    monkeypatch.setattr(cli, "_finish", lambda *args: None)
    if command == "sim":
        args = ["sim", "--frames", "1", "--serve"]
    else:
        path = tmp_path / "synthetic.esd"
        with path.open("wb") as fh:
            write_record(fh, protocol.encode(replace(frame, source=source)))
        args = ["replay", str(path), "--serve"]
    monkeypatch.setattr("sys.argv", ["lidar_mapper", *args])
    cli.main()
    assert captured[0]["source"] == delivery
    assert captured[0]["acquisition_source"] == acquisition
