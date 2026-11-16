"""CLI: serve (live from the iPhone), replay (recorded session), sim (no phone), fake-phone (test the server)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import json
import time

import numpy as np

from . import protocol
from .mapper import Mapper, save_ply
from .server import LiveMapper, map_report, read_records, run_server


def _mapper(a) -> Mapper:
    return Mapper(voxel=a.voxel, max_range=a.max_range, min_conf=a.min_conf, pose_source=a.pose,
                  map_max_range=a.map_range)


def _common(p) -> None:
    p.add_argument("--voxel", type=float, default=0.02, help="map voxel size, m")
    p.add_argument("--max-range", type=float, default=4.5, help="depth used for tracking, m")
    p.add_argument("--map-range", type=float, default=3.5, help="depth added to the map, m (LiDAR is noisier beyond ~3.5 m)")
    p.add_argument("--min-conf", type=int, default=1, help="min LiDAR confidence (0 low, 1 med, 2 high)")
    p.add_argument("--pose", choices=["imu_icp", "arkit"], default="imu_icp",
                   help="imu_icp = dark-safe (default); arkit = camera-based reference, lights on only")
    p.add_argument("--view", choices=["map", "frame", "both"], default="both")
    p.add_argument("--viewer-port", type=int, default=8765)
    p.add_argument("--capture-stall-seconds", type=float, default=3.0,
                   help="historical capture after this many seconds without successful depth ingestion (default 3)")
    p.add_argument("--viewer-points", type=int, default=20000,
                   help="points sent to the EchoSense viewer per update (default 20000; the viewer's "
                        "mapper adapter accepts up to 60000, the plain sonar decoder only 20000)")
    p.add_argument("--ply", help="write the final map to this .ply file")
    p.add_argument("--person-model", default="models/person-v0.joblib",
                   help="person detector for the viewer's Find people button (python -m lidar_mapper.detect train)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lidar_mapper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="receive the iPhone stream, map live, serve the viewer")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--record", help="append raw frames to this .esd file for replay/Databricks")
    _common(s)
    r = sub.add_parser("replay", help="map a recorded .esd session")
    r.add_argument("file")
    r.add_argument("--serve", action="store_true", help="also serve the viewer and replay in real time")
    _common(r)
    m = sub.add_parser("sim", help="map a synthetic dark room (no phone needed)")
    m.add_argument("--frames", type=int, default=120)
    m.add_argument("--serve", action="store_true", help="serve the viewer while it runs (real time, 10 fps)")
    _common(m)
    f = sub.add_parser("fake-phone", help="send synthetic frames to a running `serve`, like the iPhone would")
    f.add_argument("--url", default="ws://127.0.0.1:8770/depth")
    f.add_argument("--frames", type=int, default=120)
    f.add_argument("--fps", type=float, default=10)
    e = sub.add_parser("export", help="replay recorded .esd sessions and write Parquet + JSON for Databricks")
    e.add_argument("files", nargs="+", help=".esd recordings")
    e.add_argument("--out", default="exports", help="export folder (frames/, maps/, sessions/)")
    e.add_argument("--label", action="append", default=[], metavar="KEY=VALUE",
                   help="session label, repeatable, e.g. --label lights=off --label room=lab")
    e.add_argument("--notes", default="")
    e.add_argument("--upload", metavar="VOLUME_DIR",
                   help="also copy to this Databricks volume folder, e.g. /Volumes/workspace/<schema>/landing/lidar")
    e.add_argument("--profile", help="Databricks CLI profile for --upload")
    a = ap.parse_args(argv)
    if getattr(a, "viewer_points", 20000) != 20000:
        # Opt-in: the default stays at the tested 20,000.
        from . import server as _server
        _server.MAX_VIEWER_POINTS = max(5000, min(60000, a.viewer_points))

    if a.cmd == "export":
        from .export import export_session, upload
        labels = dict(kv.split("=", 1) for kv in a.label)
        for path in a.files:
            res = export_session(path, a.out, labels=labels, notes=a.notes)
            st = res.stats
            print(f"{res.session_id}: {st['frames']} frames, tracking {st['tracking_fraction']:.0%}, "
                  f"valid depth {st['mean_valid_fraction']:.0%}, map {st['map_points']} pts -> {a.out}/")
            if a.upload:
                for dest in upload(res, a.upload, a.profile):
                    print(f"  uploaded {dest}")
        return

    if a.cmd == "serve":
        live = LiveMapper(_mapper(a), view=a.view, record=a.record, capture_stall_s=a.capture_stall_seconds)
        _load_detector(live, a.person_model)
        try:
            asyncio.run(run_server(live, a.port, a.viewer_port))
        except KeyboardInterrupt:
            pass
        finally:
            _finish(live.mapper, a.ply)
        return

    if a.cmd == "fake-phone":
        asyncio.run(_fake_phone(a.url, a.frames, a.fps))
        return

    if a.cmd in ("replay", "sim"):
        if a.cmd == "replay":
            messages = list(read_records(a.file))
        else:
            from .sim import stream
            messages = [protocol.encode(fr) for fr in stream(a.frames)]
        live = LiveMapper(_mapper(a), view=a.view, verbose=False,
                          delivery="simulated" if a.cmd == "sim" else "recorded",
                          capture_stall_s=a.capture_stall_seconds)
        _load_detector(live, a.person_model)
        if a.serve:
            asyncio.run(_paced(live, messages, a.viewer_port))
        else:
            t0 = time.time()
            statuses = []
            for msg in messages:
                _, res = live.ingest(msg)
                statuses.append(res.status)
            dt = (time.time() - t0) / max(len(messages), 1)
            print(json.dumps({"frames": len(messages), "ms_per_frame": round(dt * 1000, 1),
                              "tracking": statuses.count("tracking"), "lost": statuses.count("lost")}))
        _finish(live.mapper, a.ply)


def _load_detector(live, path: str) -> None:
    """Person model for Find people. Missing file or missing scikit-learn only disables the button."""
    from pathlib import Path as _Path
    if not path or not _Path(path).exists():
        print(f"[detect] no person model at {path}; Find people is off")
        return
    try:
        import joblib
        live.detector = joblib.load(path)
        print(f"[detect] person model {path} loaded")
    except Exception as exc:
        print(f"[detect] could not load {path}: {exc}")


def _finish(mapper: Mapper, ply: str | None) -> None:
    print(json.dumps(map_report(mapper)))
    if ply:
        pts, counts = mapper.map.points(2)
        save_ply(ply, pts, counts)
        print(f"wrote {len(pts)} points to {ply}")


async def _paced(live: LiveMapper, messages: list[bytes], viewer_port: int, fps: float = 10) -> None:
    from websockets.asyncio.server import serve

    from .server import VIEWER_ORIGINS

    async def viewer(conn):
        await live.viewer_handler(conn)

    from .server import dense_server
    async with serve(viewer, "127.0.0.1", viewer_port, origins=VIEWER_ORIGINS + [None], compression=None), \
            dense_server(live, 8780):
        print(f"[viewer] ws://127.0.0.1:{viewer_port}/ws   full-density map: http://127.0.0.1:8780/ (Ctrl+C to stop)")
        for msg in messages:
            await asyncio.to_thread(live.ingest, msg)
            print(live.status_line(), flush=True)
            await asyncio.sleep(1 / fps)
        live.finish_capture()
        print("[replay] done; still serving the final map")
        await asyncio.Future()


async def _fake_phone(url: str, n: int, fps: float) -> None:
    from websockets.asyncio.client import connect

    from .sim import stream
    async with connect(url, max_size=None, compression=None) as ws:
        for fr in stream(n, fps):
            await ws.send(protocol.encode(fr))
            await asyncio.sleep(1 / fps)
    print(f"sent {n} frames to {url}")


if __name__ == "__main__":
    main()
