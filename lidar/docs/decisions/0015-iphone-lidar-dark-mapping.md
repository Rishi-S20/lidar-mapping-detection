# ADR-0015: Pivot to iPhone Pro LiDAR; mapping must work without light

- **Status:** accepted (Mac side tested on simulation; phone app not yet run on a device)
- **Date:** 2026-11-14
- **Area:** project + sensing (supersedes the ultrasonic sensor assumption of ADR-0001..0013 for the demo)

## Context
- The team could not source an ultrasonic sensor. Available: iPhone Pro (LiDAR) and cameras.
- The framing stays **low light**. The lead's requirement: **nothing may rely on
  visible light; mapping must work in the dark.**
- Discussed and set aside along the way (see log 2026-11-14 session 7):
  smoke (iPhone LiDAR is ~940 nm near-IR and is scattered by smoke, so it does not
  see through dense smoke), search for people under rubble (LiDAR, like sonar, sees
  only exposed surfaces), and ARKit's pose / mesh / RoomPlan / body tracking
  (all camera-based, so they degrade or fail in the dark).

## Decision
1. **Capture:** our own SwiftUI app (`lidar/ios/`), not Record3D: free, our format,
   and full access to the sensors. Default source is the **AVFoundation LiDAR
   depth camera** (no world tracking). ARKit scene depth is an optional
   reference mode, and its pose is labelled camera-based.
2. **Transport:** binary WebSocket, depth as float16 + IMU attitude/gravity per
   frame, ~10 fps (`lidar/PROTOCOL.md` v0.1). The phone drops frames rather than
   queueing them.
3. **Pose without light:** orientation from CoreMotion (gravity gives z up; gyro +
   compass give yaw), position from **point-to-plane ICP** of each depth frame
   against the map, seeded by the IMU rotation change and constant velocity.
   Frames that fit badly or disagree with the IMU by > 10° are marked lost and not
   mapped. Implemented in numpy/scipy (`lidar/mapper/`), no Open3D dependency.
4. **Map:** 3 cm voxel grid with hit counts; streamed to the existing EchoSense
   viewer through the same JSON envelope the sonar bridge uses (`kind:
   lidar_map`), so the viewer needs no change. Sessions are recorded (`.esd`) for
   replay and later Databricks ingestion; maps export to `.ply`.
5. Mac-as-origin positioning (measuring the phone's distance to the Mac) was
   rejected: one distance can't give position or orientation; Macs lack UWB;
   acoustic ranging needs several anchors. The map's origin is the phone's first pose.

## Consequences
- The sonar simulator, sonar features and Databricks sonar tables don't apply to
  the new sensor as-is. The pipeline pattern (bronze/silver/gold, MLflow,
  person bias, end-to-end diagnosis) carries over; classes and features will be
  redone for LiDAR point clouds next.
- **Unverified until a phone run:** (a) how much valid depth Apple's LiDAR
  pipeline returns in full darkness (its depth output may also use the RGB
  camera internally); (b) the camera→device axis convention
  (`geometry.CAM_TO_DEVICE`) for AVFoundation depth buffers; (c) whether the
  LiDAR depth camera delivers calibration data on every frame.
- ICP odometry drifts and can slide along featureless walls. The simulated
  walk drifts ≤ 6 cm over 1.2 m; real drift will be larger.
