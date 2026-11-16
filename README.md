# lidar-mapping-detection

An iPhone LiDAR mapper that finds people in the dark. The phone streams depth frames and
motion-sensor readings to a Mac, which builds a live 3D map without using any camera image,
then a detector looks for people in that map.

Started as a team hackathon project (EchoSense, MadHacks, November 2026). This fork keeps the
LiDAR path only and is being cleaned up.

Everything is under [`lidar/`](lidar/). Start with [lidar/README.md](lidar/README.md).

| Part | Where |
|---|---|
| iPhone capture app (Swift) | `lidar/ios/` |
| Live mapper, viewer, person detector (Python) | `lidar/mapper/` |
| Databricks pipeline and training job | `lidar/databricks/` |
| Design notes | `lidar/docs/` |

Status: the person detector is a first baseline. It finds 4 of 11 real people in held-out scans
([ADR-0019](lidar/docs/decisions/0019-lidar-person-detector-v0.md)).
