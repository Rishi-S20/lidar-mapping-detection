# ADR-0017: Label scans in the live viewer with 3D boxes; keep labels aligned with the live map

- **Status:** accepted
- **Date:** 2026-11-14
- **Area:** data pipeline (LiDAR)

## Context
The next stage (embeddings, then a person/pet/object classifier) needs labelled data.
The team is about to collect scans. Labels must say *where* each person, pet or
object is in the 3D map, not just what the scene contained.

## Decision
1. **Label right after scanning, in the full-density viewer** (`viewer.html`):
   - **Scene fields:** lights, room, scene type, notes.
   - **Object boxes:** clicked as two corners on the top-view floor plan. Box height comes
     from the map points inside, with the ceiling ignored.
   - **Classes:** person, pet, furniture, clutter, door, other. Pose is optional.
2. **Labels live in the map frame of the live session.** The live mapper drops frames
   when busy, so a later replay can produce slightly different poses and a shifted
   map, which would misalign the boxes. So the server saves, next to each recording:
   - `<scan>.live.npz`: live map snapshot + per-frame live poses
   - `<scan>.labels.json`: scene fields + boxes

   It saves them on Save labels, on Clear map, and on shutdown. `export` uses these when
   present (`map_source: live`); otherwise it replays. Frames the live mapper skipped
   are exported with `pose_status = not_processed` and the previous pose.
3. **Databricks:**
   - `silver_lidar_annotations`: one row per box.
   - `gold_lidar_labeled_points`: map points inside each box, via a range join.
   - `gold_lidar_annotation_summary`: box sizes and point support.

## Alternatives considered
- **Labels per scan only** (scene = person_lying): no location, so the person's points
  can't be separated from the furniture. Kept as scene metadata.
- **Labelling later in an external tool** (CloudCompare, Label Studio 3D): slower, a second
  coordinate convention to manage, and nothing ties it back to sessions.
- **Recomputing poses at export and ignoring the live map:** boxes drawn on the live
  map could shift by centimetres to decimetres.

## Consequences
- Boxes are axis-aligned in the map frame, so a person lying diagonally gets a loose box.
  Fine for a first dataset; rotated boxes can come later.
- Classification will treat points inside a box as positives and other map clusters
  as background/negatives.
