# Depth stream v0.1 (iPhone → Mac)

The iPhone capture app (`ios/`) sends one **binary WebSocket message per depth
frame** to the Mac mapper (`mapper/`, default `ws://<mac>:8770/depth`).
Nothing in it depends on visible light: depth comes from the LiDAR and
orientation from the IMU. ARKit's camera-based pose is optional and marked as a
reference only (ADR-0015).

## Message layout (little-endian)

| Bytes | Content |
|---|---|
| 0–3 | `uint32` header length `H` |
| 4 … 4+H | UTF-8 JSON header |
| next `w*h*2` | depth, `float16` meters, row-major, `0` or non-finite = no return |
| next `w*h` | confidence `uint8` (0 low, 1 medium, 2 high), only if `header.confidence` is true |

## Header

| Key | Type | Meaning |
|---|---|---|
| `v` | string | `"0.1"` |
| `seq` | int | Frame counter since the app started streaming |
| `t` | float | Capture time, seconds (device uptime clock, same clock as the IMU) |
| `source` | string | `"lidar"` (AVFoundation LiDAR depth camera) or `"arkit"` (ARKit scene depth) |
| `w`, `h` | int | Depth map size in pixels (sensor's native landscape orientation) |
| `fx`, `fy`, `cx`, `cy` | float | Intrinsics **in depth-map pixels** (OpenCV convention: x right, y down, z forward) |
| `confidence` | bool | Whether a confidence plane follows the depth |
| `filtered` | bool | AVFoundation temporal hole-filling on/off |
| `attitude` | `[w, x, y, z]` | CoreMotion attitude quaternion (z up; gyro + accelerometer + magnetometer; no camera) |
| `attitude_ref` | string | `magnetic_north` (yaw is the same across app restarts) or `arbitrary` (yaw resets each start). Added 2026-11-14, optional; default `arbitrary` |
| `gravity` | `[x, y, z]` | Gravity in the device frame, units of g |
| `rotation_rate` | `[x, y, z]` | rad/s, device frame |
| `user_accel` | `[x, y, z]` | g, device frame, gravity removed |
| `arkit_pose` | 16 floats or null | ARKit camera transform, column-major. **Reference only**: camera-based, fails in the dark |
| `arkit_tracking` | string or null | `normal`, `limited:<reason>`, `notAvailable` |
| `ambient_lux` | float or null | ARKit light estimate (logs how dark it was) |

Device frame (CoreMotion): x right, y toward the top of the phone, z out of the
screen, in portrait. The mapper converts camera-frame points to the device frame
with a fixed rotation (`geometry.CAM_TO_DEVICE`) and checks it against gravity.

## Versioning

Bump `v` on any breaking change and update `ios/` and `mapper/` together.
