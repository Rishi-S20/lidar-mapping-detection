# ADR-0016: LiDAR sessions into Databricks: export on the Mac, pure-Spark pipeline

- **Status:** accepted (deployed to the lead's Free Edition workspace, 2026-11-14)
- **Date:** 2026-11-14
- **Area:** data pipeline (LiDAR); supersedes ADR-0014's sonar tables as the live data path

## Context
Scans are recorded on the Mac as `.esd` files (raw frames exactly as the phone
sent them). The next stage (embeddings / classification, not started yet)
needs the scans in Databricks with quality checks, and the Databricks track is
the target. Joining scans across sessions is deprioritized.

## Decision
1. **Raw first.** The `.esd` recording stays the source of truth on the Mac. For
   Databricks, `lidar_mapper export` replays it through the mapper and writes:
   - `frames/<session>.parquet`: one row per frame. **Raw float16 depth bytes**, intrinsics,
     IMU, and the mapper's pose for that frame.
   - `maps/<session>.parquet`: map voxels seen ≥ 2 times (x, y, z, hits, segment).
   - `sessions/<session>.json`: labels (`--label lights=off`), notes, mapper settings, stats.
   Parquet rather than `.esd` so Spark reads it natively.
2. **Upload** with `--upload <volume dir>` (Databricks CLI `fs cp`) into a Unity
   Catalog volume. A job with a **file-arrival trigger** on `sessions/` refreshes the pipeline.
3. **Separate bundle in `lidar/`** (`echosense_lidar`, its own schema), owned by
   the LiDAR work. The sonar bundle stays as history.
4. **Pure Spark pipeline, no custom wheel.** Poses and maps are computed on
   the Mac, so the cluster needs no `lidar_mapper` install. (The sonar
   pipeline's first deploy failed on exactly that.) Raw depth stays in bronze,
   so a later Databricks job can recompute poses with a better mapper.
5. **Tables.**
   - Bronze: frames, maps, sessions.
   - Silver: frames with integrity checks; map points with finite and seen-twice checks;
     sessions flattened, with `lights` taken from the labels.
   - Gold: per-session summary, quality by lighting (the dark test), and a 5-second timeline.

## Consequences
- The embedding stage reads `silver_lidar_map_points` (per-session point clouds)
  and `silver_lidar_frames` (per-frame depth + pose).
- **Poses are only as good as the Mac mapper version that exported them**
  (`mapper_version`, `mapper_settings` in the session row). Re-export after a mapper change.
- About 60 MB per 1,000 frames (depth) and about 20 MB per map. Fine for Free Edition.
- **Label sessions when exporting** (`lights=on/off`, `room=…`). The four first
  recordings have no lights label, since the lead didn't record which parts were dark.
