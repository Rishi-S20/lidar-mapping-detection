"""Live server: iPhone depth stream in, 3D map out to the EchoSense viewer.

  phone  --(binary depth frames, PROTOCOL.md)-->  ws://0.0.0.0:<port>/depth
  viewer <--(sonar_frame JSON envelope 0.1)--     ws://127.0.0.1:<viewer_port>/ws

The viewer side reuses the envelope the EchoSense viewer already decodes
(websocket-bridge/PROTOCOL.md) with `meta.kind = "lidar_map"`, so no viewer
change is needed. Points are world-frame (z up), which the viewer reads as
x forward, y left, z up.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
import time
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from . import protocol
from .geometry import backproject
from .mapper import Mapper, _pack, floor_check

VIEWER_ORIGINS = ["http://127.0.0.1:4173", "http://localhost:4173", "http://127.0.0.1:5173",
                  "http://localhost:5173", "http://127.0.0.1:3000", "http://localhost:3000"]
MAX_VIEWER_POINTS = 20000  # EchoSense viewer limit
MAX_DENSE_POINTS = 800000  # viewer.html (WebGL handles this easily)
VIEWER_HTML = Path(__file__).with_name("viewer.html")
FRAME_POINTS = 2500  # current frame, drawn on top of the map


def write_record(fh, message: bytes) -> None:
    fh.write(struct.pack("<I", len(message)))
    fh.write(message)


def read_records(path: str | Path):
    with open(path, "rb") as fh:
        while True:
            head = fh.read(4)
            if len(head) < 4:
                return
            (n,) = struct.unpack("<I", head)
            yield fh.read(n)


class LiveMapper:
    """Keeps only the newest phone frame (no queue), maps it, and serves snapshots to viewers."""

    def __init__(self, mapper: Mapper, view: str = "both", record: str | None = None, verbose: bool = True,
                 delivery: str = "live", capture_stall_s: float = 3.0):
        if delivery not in ("live", "simulated", "recorded"):
            raise ValueError("Unknown mapper delivery mode")
        self.delivery = delivery
        if not math.isfinite(capture_stall_s) or capture_stall_s <= 0:
            raise ValueError("capture_stall_s must be finite and positive")
        self.capture_stall_s = capture_stall_s
        self.last_ingested_at: float | None = None  # Host monotonic seconds, never phone timestamps.
        self._next_phone = 0
        self._phones: set[int] = set()
        self._capture_phone: int | None = None
        self.capture_complete = False
        self._acquisitions: set[str] = set()
        self.mapper = mapper
        self.view = view
        self.lock = threading.Lock()
        self.latest: tuple[int, bytes] | None = None
        self.event = asyncio.Event()
        self.frame_points = np.zeros((0, 3))
        self.last = None
        self.rx = self.mapped = self.dropped = self.bad = 0
        self.seq = 0
        self.verbose = verbose
        self.record_path = Path(record) if record else None
        self.current_path = self.record_path  # recording file of the scan in progress
        self.record_fh = open(record, "ab") if record else None
        self.scan_index = 1
        self.pose_log: list[tuple] = []  # (t, seq, status, R9, t3, rmse, inliers, segment) per mapped frame
        self.labels: dict = {"scene": {}, "boxes": []}  # labels drawn in viewer.html for this scan
        self.started = time.time()
        self._disp, self._disp_size, self._disp_map = None, 0, None
        self._disp_voxel = 0.05
        self.detector = None  # optional person model (detect.py), set by --person-model
        self.detections: list[dict] = []  # person boxes from the last "Find people"
        self.person_range: dict | None = None  # distance to a detected person for the newest frame

    # ---------------------------------------------------------------- phone side
    async def phone_handler(self, conn) -> None:
        peer = conn.remote_address
        with self.lock:
            self._next_phone += 1
            phone = self._next_phone
            self._phones.add(phone)
        print(f"[phone] connected {peer}")
        try:
            async for msg in conn:
                if not isinstance(msg, bytes):
                    continue
                self.rx += 1
                if self.record_fh:
                    write_record(self.record_fh, msg)
                if self.latest is not None:
                    self.dropped += 1
                self.latest = (phone, msg)
                self.event.set()
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self._phones.remove(phone)
        print(f"[phone] disconnected {peer}")

    def ingest(self, msg: bytes, phone: int | None = None):
        """Decode + map one message (runs in a worker thread)."""
        f = protocol.decode(msg)
        with self.lock:
            if phone is not None and phone not in self._phones:
                self.dropped += 1
                return
            r = self.mapper.process(f)
            m = self.mapper
            self.pose_log.append((f.t, f.seq, r.status, r.R.reshape(-1).tolist(), r.t.tolist(), r.rmse,
                                  r.inlier_frac, len(m.segments) + m.merges + m.resets))
            # Raw recordings retain the acquisition source in each depth-frame header.
            # Unknown legacy sources must never be inferred to be live phone data.
            acquisition = "simulated" if f.source in ("sim", "simulated", "synthetic") else (
                "live" if f.source in ("lidar", "arkit") else "recorded")
            self._acquisitions.add(acquisition)
            pts = backproject(f.depth, f.K, f.confidence, self.mapper.min_conf, max_range=self.mapper.max_range,
                              stride=2)
            self.frame_points = pts @ r.R.T + r.t
            if self.detections:
                from .detect import person_range
                self.person_range = person_range(f.depth, f.K, r.R, r.t, self.detections)
            self.last = (f, r)
            self.mapped += 1
            self.last_ingested_at = time.monotonic()
            self._capture_phone = phone
            self.capture_complete = False
        return f, r

    def finish_capture(self) -> None:
        """Finite sim/replay is complete; keep serving its final historical map."""
        with self.lock:
            self.capture_complete = True

    def _capture_status(self) -> dict:
        # Called under the snapshot lock. Sending a snapshot never refreshes ingestion.
        age = None if self.last_ingested_at is None else max(0.0, time.monotonic() - self.last_ingested_at)
        mode = "replay" if self.delivery == "recorded" else (
            "simulated" if self.acquisition_source == "simulated" else
            "live" if self.acquisition_source == "live" else "unknown")
        if self.capture_complete and self.delivery != "live":
            state = "complete"
        elif self.delivery == "live" and not self._phones:
            state = "disconnected"
        elif self.delivery == "live" and self._capture_phone not in self._phones:
            state, age = "waiting", None
        elif age is None:
            state = "waiting"
        else:
            state = "stalled" if age > self.capture_stall_s else "active"
        return {"mode": mode, "state": state, "age_s": None if age is None else round(age, 3),
                "stall_after_s": self.capture_stall_s}

    @property
    def acquisition_source(self) -> str:
        """Conservative attribution of the accumulated map, not just its last frame."""
        if self.delivery == "simulated" or "simulated" in self._acquisitions:
            return "simulated"
        return "live" if self._acquisitions == {"live"} else "recorded"

    async def worker(self) -> None:
        while True:
            await self.event.wait()
            self.event.clear()
            pending, self.latest = self.latest, None
            if pending is None:
                continue
            try:
                phone, msg = pending
                await asyncio.to_thread(self.ingest, msg, phone)
            except Exception as e:  # bad frame: count it, keep going
                self.bad += 1
                if self.bad <= 5:
                    print(f"[phone] rejected frame: {e}")

    # ---------------------------------------------------------------- viewer side
    def _display_map(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The whole map, evenly thinned to <= n points (coarser grid as the map grows).

        A random sample of a 500k-voxel map showed ~3% of it and made the map look like it
        wasn't accumulating (first phone runs, 2026-11-14). Cached; rebuilt when the map grows 2%.
        """
        size = len(self.mapper.map)
        if self._disp is not None and abs(size - self._disp_size) <= 0.02 * max(self._disp_size, 1) \
                and self._disp_map is self.mapper.map:
            return self._disp
        p, c = self.mapper.map.points(2)
        voxel = max(0.05, self._disp_voxel / 1.25)  # the map only grows: start near last time's size
        while True:
            _, idx = np.unique(_pack(np.floor(p / voxel).astype(np.int64)), return_index=True)
            if len(idx) <= n or len(p) == 0:
                break
            voxel *= 1.25
        self._disp_voxel = voxel
        self._disp, self._disp_size, self._disp_map = (p[idx], c[idx]), size, self.mapper.map
        return self._disp

    def snapshot(self) -> str:
        with self.lock:
            rows = []
            if self.view in ("map", "both"):
                n = MAX_VIEWER_POINTS - (FRAME_POINTS if self.view == "both" else 0)
                p, c = self._display_map(n)
                rows += [[round(float(x), 3), round(float(y), 3), round(float(z), 3), float(min(k, 50))]
                         for (x, y, z), k in zip(p, c)]
            if self.view in ("frame", "both") and len(self.frame_points):
                fp = self.frame_points
                if len(fp) > FRAME_POINTS:
                    fp = fp[np.random.default_rng(self.seq).choice(len(fp), FRAME_POINTS, replace=False)]
                rows += [[round(float(x), 3), round(float(y), 3), round(float(z), 3), 100.0] for x, y, z in fp]
            acquisition = self.acquisition_source
            meta = {"format_version": "0.2", "source": acquisition, "sensor_id": "iphone-lidar",
                    "tilt_deg": 0.0, "kind": "lidar_map", "view": self.view,
                    "capture": self._capture_status()}
            t = float(self.last[0].t) if self.last else 0.0
        self.seq += 1
        return json.dumps({
            "protocol_version": "0.1", "type": "sonar_frame", "sequence": self.seq, "timestamp": t,
            "source": self.delivery, "acquisition_source": acquisition,
            "frame": {"format_version": "0.2",
                      "meta": meta,
                      "points": rows[:MAX_VIEWER_POINTS]},
            "detections": None,
        })

    async def viewer_handler(self, conn, fps: float = 5.0) -> None:
        print("[viewer] connected")
        try:
            while True:
                await asyncio.wait_for(conn.send(self.snapshot()), timeout=5)
                await asyncio.sleep(1 / fps)
        except (ConnectionClosed, asyncio.TimeoutError):
            pass
        print("[viewer] disconnected")

    # ---------------------------------------------------------------- dense viewer (viewer.html)
    def dense_snapshot(self, send_map: bool) -> bytes:
        """Binary message for viewer.html: JSON header + float32 xyz (map, live frame, trajectory)."""
        with self.lock:
            m = self.mapper
            mp = np.zeros((0, 3), np.float32)
            if send_map:
                p, _ = m.map.points(2)
                if len(p) > MAX_DENSE_POINTS:  # thin evenly, never randomly
                    voxel = m.map.voxel
                    while len(p) > MAX_DENSE_POINTS:
                        voxel *= 1.25
                        _, idx = np.unique(_pack(np.floor(p / voxel).astype(np.int64)), return_index=True)
                        p = p[idx]
                mp = p.astype(np.float32)
            fp = self.frame_points.astype(np.float32)
            if len(fp) > 30000:
                fp = fp[::int(np.ceil(len(fp) / 30000))]
            tr = np.asarray(m.trajectory[-5000:], np.float32).reshape(-1, 3)
            f, r = self.last if self.last else (None, None)
            hdr = {"n_map": int(len(mp)) if send_map else -1, "n_frame": int(len(fp)), "n_traj": int(len(tr)),
                   "status": r.status if r else "waiting", "voxels": len(m.map),
                   "valid": f.valid_fraction if f else None,
                   "pose": [float(x) for x in r.t] if r else None, "segments": len(m.segments),
                   "person_range": self.person_range}
        hb = json.dumps(hdr).encode()
        pad = (4 - (4 + len(hb)) % 4) % 4
        return (struct.pack("<I", len(hb)) + hb + b"\0" * pad + mp.tobytes() + fp.tobytes() + tr.tobytes())

    async def dense_handler(self, conn, fps: float = 4.0) -> None:
        print("[map viewer] connected")
        state = {"last_size": -1, "last_sent": 0.0}

        async def labels_msg(saved=None):
            await conn.send(json.dumps({"type": "labels", **self.labels, "saved": saved,
                                        "scan": self.current_path.name if self.current_path else None}))

        async def commands():  # JSON commands from viewer.html buttons
            async for msg in conn:
                try:
                    cmd = json.loads(msg) if isinstance(msg, str) and msg.startswith("{") else {"cmd": msg}
                except ValueError:
                    continue
                if cmd.get("cmd") == "clear":
                    await asyncio.to_thread(self.clear)
                    state["last_size"] = -1  # push the (now empty) map right away
                    await labels_msg()
                elif cmd.get("cmd") == "labels":
                    saved = await asyncio.to_thread(self.set_labels, cmd.get("scene", {}), cmd.get("boxes", []))
                    await labels_msg(saved)
                elif cmd.get("cmd") == "find":
                    await conn.send(json.dumps({"type": "detections", **await asyncio.to_thread(self.find_people)}))

        await labels_msg()  # restore this scan's labels after a page reload

        cmd_task = asyncio.create_task(commands())
        try:
            while True:
                size, now = len(self.mapper.map), time.time()
                send_map = state["last_size"] < 0 or (size != state["last_size"] and now - state["last_sent"] > 1.0)
                msg = await asyncio.to_thread(self.dense_snapshot, send_map)
                if send_map:
                    state["last_size"], state["last_sent"] = size, now
                await asyncio.wait_for(conn.send(msg), timeout=10)
                await asyncio.sleep(1 / fps)
        except (ConnectionClosed, asyncio.TimeoutError):
            pass
        finally:
            cmd_task.cancel()
        print("[map viewer] disconnected")

    # ---------------------------------------------------------------- status
    def status_line(self) -> str:
        if not self.last:
            return f"[map] waiting for frames (received {self.rx}, rejected {self.bad})"
        f, r = self.last
        lux = f"{f.ambient_lux:.0f} lux" if f.ambient_lux is not None else "lux n/a"
        return (f"[map] rx {self.rx} mapped {self.mapped} dropped {self.dropped} | {r.status:8s} "
                f"pos ({r.t[0]:+.2f},{r.t[1]:+.2f},{r.t[2]:+.2f}) m | valid depth {f.valid_fraction:.0%} | "
                f"rmse {r.rmse * 100:.1f} cm | grav err {r.gravity_err_deg:.1f} deg | {lux} | "
                f"map {len(self.mapper.map)} voxels")

    async def status(self, every: float = 1.0) -> None:
        while True:
            await asyncio.sleep(every)
            if self.verbose:
                print(self.status_line(), flush=True)

    # ---------------------------------------------------------------- per-scan sidecars
    def save_sidecars(self) -> list[str]:
        """Save this scan's live map, live poses and labels next to its recording.

        Labels are drawn on the *live* map, whose poses can differ slightly from a later replay
        (the live mapper drops frames when busy). Export uses these files so boxes stay aligned.
        """
        if not self.current_path:
            return []
        out = []
        with self.lock:
            pts, cnt = self.mapper.map.points(2)
            log = list(self.pose_log)
            labels = json.loads(json.dumps(self.labels))
        if log:
            live = self.current_path.with_suffix(".live.npz")
            np.savez_compressed(
                live, map_points=pts.astype(np.float32), map_hits=cnt.astype(np.int32),
                t=np.array([r[0] for r in log]), seq=np.array([r[1] for r in log], np.int64),
                status=np.array([r[2] for r in log]), R=np.array([r[3] for r in log], np.float64),
                pos=np.array([r[4] for r in log], np.float64), rmse=np.array([r[5] for r in log], np.float64),
                inliers=np.array([r[6] for r in log], np.float64), segment=np.array([r[7] for r in log], np.int64),
                voxel=np.array(self.mapper.map.voxel))
            out.append(str(live))
        if labels["scene"] or labels["boxes"]:
            lab = self.current_path.with_suffix(".labels.json")
            labels["scan_file"] = self.current_path.name
            labels["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            lab.write_text(json.dumps(labels, indent=2))
            out.append(str(lab))
        return out

    def set_labels(self, scene: dict, boxes: list) -> list[str]:
        clean = []
        for b in boxes[:200]:
            lo, hi = [float(v) for v in b["min"]], [float(v) for v in b["max"]]
            if len(lo) != 3 or len(hi) != 3 or any(h <= l for l, h in zip(lo, hi)):
                continue
            clean.append({"label": str(b.get("label", "other"))[:40], "pose": str(b.get("pose", ""))[:40],
                          "min": lo, "max": hi, "note": str(b.get("note", ""))[:200]})
        self.labels = {"scene": {str(k)[:40]: str(v)[:200] for k, v in (scene or {}).items()}, "boxes": clean}
        saved = self.save_sidecars()
        print(f"[labels] {len(clean)} boxes, scene {self.labels['scene']} -> {saved}", flush=True)
        return saved

    def find_people(self) -> dict:
        """Run the person detector on the current map. Geometry only, so it does not depend on light."""
        if self.detector is None:
            return {"boxes": [], "note": "no person model loaded (start the mapper with --person-model)"}
        from .detect import build_grid, detect
        with self.lock:
            points, _ = self.mapper.map.points(2)
        if len(points) < 5000:
            return {"boxes": [], "note": "map too small to search yet"}
        found = detect(build_grid(points), self.detector)
        with self.lock:
            self.detections, self.person_range = found, None
        print(f"[detect] {len(found)} person box(es): {[(d['score'], d['height_m']) for d in found]}", flush=True)
        return {"boxes": found, "note": f"{len(found)} found" if found else "nobody found"}

    def clear(self) -> str:
        """Wipe the map and start a fresh one; recording continues in a new file (<name>-scan2.esd, ...)."""
        self.save_sidecars()  # keep the finished scan's live map/poses/labels
        with self.lock:
            self.pose_log = []
            self.labels = {"scene": {}, "boxes": []}
            self.detections, self.person_range = [], None
            m = self.mapper
            m.reset()
            m.segments, m.merges, m.relocalizations = [], 0, 0
            m.need_reloc, m.last_seq, m.prev_time = False, -1, None
            self.frame_points = np.zeros((0, 3))
            self.last = None
            self._disp, self._disp_size, self._disp_map = None, 0, None
            self.scan_index += 1
            note = "map cleared"
            if self.record_path:
                self.record_fh.close()
                new = self.record_path.with_name(f"{self.record_path.stem}-scan{self.scan_index}{self.record_path.suffix}")
                self.record_fh = open(new, "ab")
                self.current_path = new
                note += f"; now recording to {new}"
        print(f"[map] {note}", flush=True)
        return note

    def close(self) -> None:
        try:
            self.save_sidecars()
        finally:
            if self.record_fh:
                self.record_fh.close()


def dense_server(live: LiveMapper, port: int = 8780):
    """http://127.0.0.1:<port>/ serves viewer.html; ws://127.0.0.1:<port>/map streams the full map."""
    from http import HTTPStatus

    def http(connection, request):
        if request.path in ("/", "/index.html"):
            resp = connection.respond(HTTPStatus.OK, VIEWER_HTML.read_text())
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp
        if request.path != "/map":
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        return None  # WebSocket upgrade

    async def handler(conn):
        await live.dense_handler(conn)

    return serve(handler, "127.0.0.1", port, process_request=http, max_size=512 * 1024, compression=None)


async def run_server(live: LiveMapper, port: int = 8770, viewer_port: int = 8765, host: str | None = None,
                     viewer_fps: float = 5.0, origins: list[str] | None = None, map_port: int = 8780) -> None:
    async def phone(conn):
        await live.phone_handler(conn)

    async def viewer(conn):
        await live.viewer_handler(conn, viewer_fps)

    tasks = [asyncio.create_task(live.worker()), asyncio.create_task(live.status())]
    async with serve(phone, host, port, max_size=4 * 1024 * 1024, max_queue=2, compression=None), \
            serve(viewer, "127.0.0.1", viewer_port, origins=(origins or VIEWER_ORIGINS) + [None],
                  max_size=1024, compression=None), \
            dense_server(live, map_port):
        # host=None listens on every IPv4 *and* IPv6 address. An iPhone hotspot can be IPv6-only
        # (the Mac then shows a 192.0.0.x CLAT address the phone can't reach), see lidar/README.md.
        print(f"[server] phone -> ws://<this Mac>:{port}/depth (all IPv4 + IPv6 addresses)   "
              f"EchoSense viewer -> ws://127.0.0.1:{viewer_port}/ws")
        print(f"[server] full-density map viewer -> http://127.0.0.1:{map_port}/")
        try:
            await asyncio.Future()
        finally:
            for t in tasks:
                t.cancel()
            live.close()


def map_report(mapper: Mapper) -> dict:
    pts, _ = mapper.map.points(2)
    return {"voxels": len(mapper.map), "points_seen_twice": int(len(pts)), "floor": floor_check(pts)}
