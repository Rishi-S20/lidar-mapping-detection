# 2026-11-14 (session 7) — Pivot to iPhone LiDAR, dark-safe live mapping

_Decision record: [ADR-0015](../decisions/0015-iphone-lidar-dark-mapping.md).
Code: `lidar/` at the repo root (still on branch `databricks-pipeline`, uncommitted)._

## How we got here
1. **No ultrasonic sensor available.** Cameras were suggested instead. Options
   discussed: keep sonar via the laptop speaker/mic (Ryan's `acoustic/`) or a
   cheap HC-SR04; pre-incident mapping; "last seen" occupant locations from CCTV.
2. **Lead chose iPhone Pro LiDAR.** Noted that Apple's RoomPlan / ARKit already
   classify walls, doors and furniture, so our model should target what Apple
   doesn't (people, pets, debris, hazards).
3. **Search and rescue for people/pets in rubble:** LiDAR can't see through
   rubble, only partly exposed bodies; colour + depth fusion would help. Then:
4. **"Does LiDAR work past smoke?"** No for dense smoke: ~940 nm near-IR is
   strongly scattered by 0.1-1 µm smoke particles, iPhone LiDAR is low power,
   and ARKit tracking needs the camera. Thermal (LWIR) and mmWave radar do see
   through smoke.
5. **Framing must be low light.** Options: night fall detection (depth-only,
   privacy friendly), dark-building search, caves/tunnels. Lead: "first build it
   out": live feed and mapping first.
6. **Swift app vs Record3D:** chose our own Swift app (free, full sensor access,
   our format). Record3D's live streaming is a paid in-app purchase.
7. **"We can't have anything rely on light."** This rules out ARKit's pose,
   mesh, RoomPlan and body tracking. Mac-as-origin distance positioning was
   considered and rejected (one distance ≠ 6-DoF pose; no UWB on Macs).
   Chosen: IMU orientation + LiDAR ICP for position.

## Built
- `lidar/PROTOCOL.md`: binary depth stream v0.1.
- `lidar/ios/`: SwiftUI app. AVFoundation `builtInLiDARDepthCamera` depth
  (float16) + CoreMotion attitude at 100 Hz; optional ARKit reference mode;
  WebSocket client that drops frames instead of queueing; dark UI. Compiles for
  iOS 17 (`xcodebuild -sdk iphoneos CODE_SIGNING_ALLOWED=NO`), **not yet run on a phone**.
- `lidar/mapper/`: Python mapper (numpy/scipy/websockets). Point-to-plane ICP
  with an IMU rotation prior, packed-key voxel map, live server (phone in,
  viewer out with the existing envelope), recording/replay, PLY export,
  synthetic dark-room simulator, 8 tests.

## Trial and error
- **Simulator pitch sign was backwards:** "tilt down" looked at the ceiling. Fixed in `sim.device_rotation`.
- **Drift 16 cm over a 1.2 m walk** at 6 cm ICP voxels → 4.5 cm at 4 cm voxels, but 280 ms/frame.
- **Speed:** profiled. ICP queries and target normals dominated. Changes:
  packed 1-D voxel keys + searchsorted merge, rebuild the ICP target every 2
  frames, cap ICP source at 4,000 points, parallel KD-tree queries.
  Result: **~55 ms/frame, ≤ 6 cm drift, 79/80 frames tracked**.
- The floor check first averaged in low furniture (reported -1.24 m for a -1.30 m
  floor); it now uses only the lowest 8 cm band.
- CoreMotion's quaternion convention (device→reference or reverse) is
  ambiguous in Apple's docs, so the mapper picks whichever one sends measured
  gravity to "down". Tested with both conventions.

## Next
1. First phone run: valid-depth % with lights on vs fully dark; floor tilt in the map report.
2. If AVFoundation depth is poor in the dark, try the ARKit source's depth (pose ignored).
3. Then: person / pet / obstacle classification on the LiDAR point clouds,
   and recorded sessions into the Databricks pipeline.

## First phone run (lights on, same Wi-Fi)
- Installed via `xcodebuild` + `devicectl` with the lead's free personal team (bundle id
  overridden on the command line, not committed). Developer Mode had to be enabled first.
- Phone (iPhone18,2) streamed **292 frames, 320x240 float16 depth, calibration on
  every frame**, ~10 fps, depth 0.26-3.7 m, 0 frames dropped or rejected.
- **Live map stayed empty.** The first ~6 frames had no depth (sensor warm-up), then
  ~20 frames saw a surface 12-22 cm away (desk/hand). The mapper started its map from
  those frames (75 voxels) and then marked every later frame lost. Fix: don't start
  the map until a frame has ≥ 3,000 points, is ≥ 1 m across, and has median depth
  ≥ 0.5 m; reset if the map is still too small after 5 lost frames.
- **Replay of the same recording after the fix:** 232/292 tracking, 32 lost, 27
  waiting; median fit 2.0 cm, IMU disagreement 0.7°; map 7.5 x 4.4 x 3.2 m;
  **floor tilt 3.2°**, which confirms the `CAM_TO_DEVICE` axis convention
  for AVFoundation depth. ~70 ms/frame on 320x240.
- Lost frames cluster where only ~30% of points overlap the map (new area / fast pan).
- Not yet tested: full darkness.

## Second phone run: "the map isn't keeping what I scanned"
Recording `light-vs-dark.esd` (908 frames, 3 app sessions). Diagnosis by replay:
1. **Tracking dropped when panning into new areas.** Only ~30% of a frame overlapped
   the map and the rule required 35%, so from 51 s every frame was rejected (not
   mapped). Lowered to 20% plus ≥ 300 matched points.
2. **App restarts broke tracking for good.** CoreMotion's `xArbitraryCorrectedZVertical`
   resets yaw on every start; the mapper kept the old heading/position. Added:
   session-restart detection (seq reset or > 1.5 s gap), **relocalization** (yaw ×
   position search, ICP-refine the best 6), and when that fails, **map segments**
   that are merged back once they overlap (`Segment`, `_try_merge`). Tested on
   simulated restarts (relocalize and segment→merge both pass).
3. **The viewer showed a random ~3% of the map**, so it looked like it wasn't
   accumulating. Now the whole map, evenly thinned to ≤ 17,500 points (adaptive grid,
   cached, ~35 ms), plus 2,500 points of the live frame.
4. App now uses **`xMagneticNorthZVertical`** (`attitude_ref` header field) so heading
   survives restarts; relocalization then searches only ±20°.
- Replay after 1-3: 853/908 frames mapped (was 345). The 3 sessions did **not**
  merge: best overlap 8-11% even with a wide search. They mostly covered different
  views. Maps built from IMU rotation alone show the same angled walls as the ICP
  maps, so those walls are likely real; ICP-vs-IMU heading agrees within 2° by
  session end (max transient 19° in session 2).
- Lesson: scanning mostly by turning in place gives little overlap between views.
  Keep the app streaming (don't restart); sweep slowly and revisit areas.

## Third run (`scan3.esd`): "much better, but a little scattered"
- Causes, measured on the recording: (1) the EchoSense viewer's 20,000-point cap
  showed ~1 in 20 map points; (2) ~10% of depth was beyond 4.9 m, where iPhone
  LiDAR is noisy; (3) only every 2nd depth point went into the map; (4) ~1% of
  pixels are edge "flying pixels".
- Changes: edge-pixel filter (drop pixels whose depth jumps > 5% vs a neighbour),
  map only depth ≤ 3.5 m (`--map-range`; tracking still uses ≤ 4.5 m), all depth
  points into the map, 2 cm voxels. Speed fixes so it still runs live: packed 1-D
  voxel keys in `voxel_downsample` (np.unique axis=0 was 15 s of 33 s), no
  per-query thread pools, ICP target rebuilt every 3 frames, 3,000 ICP points.
- Result on scan3: **2.2x the map points (294k → 637k), stray specks 3.9% →
  0.7%**, 76 ms/frame (~13 fps), same tracking (682/706). Trying 4 frames /
  2,500 points instead was faster but smeared the map (1.28M points, 66 lost).
- New **full-density viewer** at `http://127.0.0.1:8780/` (Three.js, served by the
  mapper, binary float32 stream) shows the whole map. The EchoSense viewer is
  unchanged (other team's code).

## LiDAR data pipeline into Databricks (ADR-0016)
- Lead: deprioritize joining sessions; save scans and feed them to Databricks
  (no classification yet).
- Built `lidar_mapper export` (replay `.esd` → `frames/*.parquet` with raw float16
  depth + IMU + pose, `maps/*.parquet`, `sessions/*.json` with labels) and `--upload`
  (Databricks CLI). New bundle `lidar/databricks.yml`: schema, landing volume,
  pure-Spark Lakeflow pipeline (bronze/silver/gold with expectations), and an ingest job
  with a file-arrival trigger.
- Exported the 4 real recordings: 2,397 frames, 6.3M map points; 152 MB frames, 80 MB maps.
- **Bug found locally:** session JSON parsed to all-NULL rows. `option("wholetext")`
  is silently ignored; the option is `wholeText`. Caught by running the pipeline's
  expression strings in local Spark against the real exports.
- **Bug found on Databricks:** an expectation used `export_version`, which wasn't
  selected. Added it.
- **Pipeline run succeeded** (update bce112b2). Row counts: bronze/silver frames 2,397
  (no frame dropped by expectations), map points 6,309,822, sessions 4. Gold summary
  per session: tracked 84-97%, valid depth 84-91%, median ICP fit 1.7-2.1 cm,
  floor tilt 0.5-4.2°. All sessions labelled `lights=unknown`. scan4 has 3 map
  segments despite 0 app restarts, so tracking was lost mid-scan twice and the
  segments didn't merge.

## Connection trouble, viewer fixes, labelling (late session)
- **Phone couldn't reach the Mac:** venue Wi-Fi (10.140/15) isolates clients. On the
  iPhone hotspot the Mac showed **192.0.0.3/32**, a macOS CLAT address because the
  hotspot was **IPv6-only** (T-Mobile). Fix: the mapper listens on all IPv4 + IPv6
  addresses (`host=None`). The phone uses `ws://[<Mac IPv6>]:8770/depth`.
- **Dense viewer showed only a corner:** the canvas had no CSS size, so at 2x DPI it rendered
  at double size. Fixed with 100vw/100vh. Also: re-frames as the map grows until the user
  moves the camera, floor-level grid, 3D/Top buttons.
- **Clear map & start new scan** button: wipes the map and rotates the recording file
  (`-scan2.esd`, …).
- **Labelling (ADR-0017):** scene fields + 2-click top-view boxes in the viewer. Live map
  and live poses are saved next to each recording (`.live.npz`, `.labels.json`) so export
  keeps boxes aligned. Pipeline tables: `silver_lidar_annotations`,
  `gold_lidar_labeled_points`, `gold_lidar_annotation_summary`. Tested end to end on a
  simulated session (local Spark: 3,691 person + 4,420 furniture points inside boxes)
  and in the browser (box drawn, height auto 0.35 m, saved). 14 tests pass.
- Guide for the team: `lidar/DATA_COLLECTION.md`.

## 2026-11-15: "wavy noise drifting in the room"
Lead's screenshot: a person standing in an emptier area, wispy sheets in open space.
Diagnosis on `data-day1-scan4` (and scan7, scan10):
- Not speckle: stray points ~0.4%. Single frames have a thin floor (1.4 cm RMS).
- **Tilt-to-gravity prior** in ICP (roll/pitch held to the IMU, yaw free): small effect
  (scan7 floor 6.2 → 4.3 cm, scan4/10 about unchanged). Kept as the default (`tilt_prior_weight=5`).
- **Focal length is not the cause:** scaling fx by 0.85-1.16 made nothing better than 1.0.
- **Side slices showed the real cause:** floor and ceiling are thin lines, but wisps/streaks
  float in free space, and the standing person was a solid smeared blob. Those are
  grazing-angle and along-ray depth artifacts, each seen from one viewpoint.
- Fix: **grazing-angle filter** (drop pixels whose surface is > ~75° to the ray) +
  **free-space carving** (2,000 rays every 2nd frame; voxels seen through more often than
  hit are hidden). Result: the person becomes a clean outline, wisps reduced, about 10% fewer
  map points, +13 ms/frame (~94-103 ms). Streaks in areas seen from only one side remain
  (nothing to carve them with).
- Integration gates (skip frames with rmse > 2.2 cm or tilt > 1.5°) had no effect;
  left off.
- Better scans: view each area from 2+ directions so carving can remove its artifacts.

## 2026-11-15: synthetic scans to finish the checklist (ADR-0018)

- **Real scans:** 13 labelled in `data-day1b` (4 empty, 2 standing, 3 sitting, 4 lying). scan9 and
  scan20 were broken by app restarts mid-scan and are left out of the export.
- **Synthetic scans:** `lidar_mapper synth` inserted the 17 missing checklist scenes into real
  empty-room scans 10, 13 and 14: 2 standing, 2 lying, 3 crouching, 4 partly covered, 3 pet,
  3 mixed. Target surface noise is 1.3 cm against about 2 cm for real floors.
- **Box check:** 0–11% of the points in each box were already in the base room. One cat scene
  was regenerated because it landed against a wall (41%).
- **Limits found:**
  - The rooms are cluttered, so a lying person plus clutter does not fit; mixed scenes use
    sitting and crouching people.
  - Only 4 of 17 synthetic scans are lights-on, because the only lights-on empty base (scan13)
    is 10 s long.
  - All bases have Hole filling on.
- **Databricks:** 11 real + 17 synthetic sessions uploaded; `data_origin` column added
  (`real`/`synthetic`).
