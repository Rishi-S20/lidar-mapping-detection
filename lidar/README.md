# lidar/ — live 3D mapping from iPhone LiDAR, in the dark

The project's sensor is now the **iPhone Pro LiDAR** (the ultrasonic sensor
couldn't be sourced). Requirement from the team: **nothing may depend on visible
light**, so mapping uses only LiDAR depth and the IMU (gyro, accelerometer,
compass). See [ADR-0015](docs/decisions/0015-iphone-lidar-dark-mapping.md).

```
iPhone (ios/)                          Mac (mapper/)                         Browser
LiDAR depth 10 fps + IMU attitude  ->  depth -> 3D points                ->  http://127.0.0.1:8780/
binary WebSocket (PROTOCOL.md)         IMU rotation + ICP alignment          live map, labelling,
ws://<mac>:8770/depth                  voxel map, record .esd, export .ply    Find people
```

> Status 2026-11-14: **first real phone run done (lights on).** The phone streamed
> 320x240 depth + IMU at ~10 fps. After a fix to how the map starts, the recording
> maps with 232/292 frames tracked, 2 cm median fit, floor level to 3°, room
> 7.5 x 4.4 x 3.2 m. **Still to test: full darkness.** On a simulated room: 79/80
> frames, ≤ 6 cm drift over 1.2 m.

## First run

### 1. Mac: start the mapper

```bash
cd lidar/mapper
/usr/local/bin/python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m lidar_mapper serve --record recordings/session1.esd --ply recordings/session1.ply
```

It prints a status line every second: tracking state, position, % valid depth,
fit error, gravity error, map size. Ctrl+C writes the `.ply` (opens in MeshLab,
CloudCompare, Blender).

### 2. Connect the iPhone to the Mac

Recommended: **iPhone Personal Hotspot over USB** (venue Wi-Fi often blocks
phone-to-laptop traffic). Turn on Personal Hotspot, plug in the cable; the Mac
is then usually `172.20.10.2` (check with `ipconfig getifaddr en7` or
System Settings → Network). Same Wi-Fi also works if the network allows it:
use the Mac's Wi-Fi IP.

### 3. iPhone: build and run the capture app (one-time setup)

1. Open `lidar/ios/EchoSenseCapture.xcodeproj` in Xcode.
2. iPhone: Settings → Privacy & Security → **Developer Mode** on (restart).
3. Xcode → target EchoSenseCapture → Signing & Capabilities → **Team**: your
   Apple ID (a free personal team works; apps then expire after 7 days). If the
   bundle id is taken, change `com.echosense.lidarcapture` to something unique.
4. Pick your iPhone as the run destination, press Run. First launch: allow
   camera (needed for the LiDAR sensor; no images are sent) and local network.
5. In the app: Mac URL `ws://172.20.10.2:8770/depth`, depth source **LiDAR
   only**, Start streaming.

### 4. View the map

Open `http://127.0.0.1:8780/` while `serve` (or `replay --serve` / `sim --serve`) runs. It shows
the whole map (up to 800k points, coloured by height), the live frame in white, and the phone's
path. The same page has the labelling panel and the **Find people in this map** button.

## Collecting labelled data

See **[DATA_COLLECTION.md](DATA_COLLECTION.md)**: what to scan, the per-scan routine
(**Clear map** → scan → **Label this scan** panel → boxes → **Save labels**), and
export. Why it works this way: [ADR-0017](docs/decisions/0017-scan-labelling.md).

## Send scans to Databricks

```bash
cd lidar/mapper
# replay a recording, write Parquet + JSON, and upload to the landing volume
.venv/bin/python -m lidar_mapper export recordings/scan4.esd --label lights=on --label room=lab \
    --upload /Volumes/workspace/dev_<you>_echosense_lidar/landing/lidar
```

The upload triggers the `echosense-lidar-ingest` job, which refreshes the pipeline.
Tables in `workspace.dev_<you>_echosense_lidar`:

- **Bronze:** `bronze_lidar_frames`, `bronze_lidar_maps`, `bronze_lidar_sessions`
- **Silver:** `silver_lidar_frames` (raw depth + pose), `silver_lidar_map_points`
  (input for embeddings), `silver_lidar_sessions`
- **Gold:** `gold_lidar_session_summary`, `gold_lidar_quality_by_lights`, `gold_lidar_timeline`

Bundle: `lidar/databricks.yml` (deploy from `lidar/`: `databricks bundle deploy`).
Why: [ADR-0016](docs/decisions/0016-lidar-lakehouse.md).

## Without a phone

```bash
.venv/bin/python -m lidar_mapper sim --frames 120 --serve      # simulated dark room -> viewer
.venv/bin/python -m lidar_mapper serve & .venv/bin/python -m lidar_mapper fake-phone   # full WebSocket path
.venv/bin/python -m lidar_mapper replay recordings/session1.esd --ply out.ply          # re-map a recording
.venv/bin/python -m pytest
```

## How position is found without light

- **Orientation:** CoreMotion attitude (reference frame `xArbitraryCorrectedZVertical`:
  z from gravity, yaw from gyro + compass). No camera.
- **Position:** each depth frame is aligned to the map built so far with
  point-to-plane ICP, starting from the IMU rotation change and the last
  velocity. Accepted only if the fit is good (RMSE ≤ 4 cm, ≥ 35% inliers) and
  agrees with the IMU within 10°; otherwise the frame is marked `lost`, the
  position is held, and the frame is not added to the map.
- **Map:** 3 cm voxels with hit counts; the viewer shows voxels seen ≥ 2 times.
- **Origin:** where the phone was on the first frame; z up.

Known weak spots: flat featureless walls and long corridors (ICP can slide
along them), fast motion (keep it slow), and range beyond ~4 m (noisy).

## Modes in the app

| Source | What it uses | Use for |
|---|---|---|
| **LiDAR only** (default) | AVFoundation LiDAR depth camera + IMU | Everything; dark-safe |
| ARKit (reference) | ARKit scene depth + confidence + ARKit pose + light level | Lights-on comparison only. Run the Mac with `--pose arkit` to map with ARKit's camera pose and compare against `imu_icp` |

## Files

| Path | What |
|---|---|
| `PROTOCOL.md` | Binary depth stream v0.1 (iPhone → Mac) |
| `ios/` | SwiftUI capture app (AVFoundation LiDAR, CoreMotion, ARKit reference mode, WebSocket) |
| `mapper/src/lidar_mapper/` | `protocol`, `geometry` (frames, rotations), `icp`, `mapper` (pose + voxel map), `server` (phone + viewer sockets), `sim` (synthetic dark room), CLI in `__main__` |
| `mapper/tests/` | Geometry, protocol round trip, mapping accuracy on the simulated walk |
