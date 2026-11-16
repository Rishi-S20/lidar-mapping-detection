# ADR-0018: Fill the scan checklist with synthetic targets inserted into real empty-room scans

- **Status:** superseded. The generator was removed on 2026-11-16: synthetic scans did not improve detection on real scans ([ADR-0019](0019-lidar-person-detector-v0.md)). Kept as a record of what was tried.
- **Date:** 2026-11-15
- **Area:** data pipeline (LiDAR)

## Context
On 2026-11-15 the team recorded 13 labelled real scans: empty rooms, standing, sitting and lying.
The [checklist](../../DATA_COLLECTION.md) still needs 17 more: crouching, partly covered,
pets and mixed clutter, plus extra standing and lying scans. Real pets and staged rubble aren't
available at the venue. We want a full dataset so the pipeline and the next stage
(embeddings/classifier) can be built end to end now.

## Decision
`python -m lidar_mapper synth <empty-room .esd ...>` (`lidar/mapper/src/lidar_mapper/synth.py`)
builds each missing scene **on top of a real empty-room recording**:

1. **Real parts, reused.** The base recording supplies:
   - the phone's real path (the live poses in `<scan>.live.npz`)
   - real room depth, dropouts and IMU
   - the lights label
2. **The target.** Simple shapes (capsules, ellipsoids, boxes) for a person (standing,
   crouching, sitting, lying on the back, front, side or curled), a dog or cat, a blanket or
   cardboard box over a person, and clutter. Lying bodies get a soft "clothing" envelope so they
   map as one low mound, like the real lying scans do, rather than separate limbs.
3. **Placement.** On a free patch of real floor (floor seen under it, nothing standing on it)
   that the real scan path looked at from 1–3.2 m.
4. **Rendering.**
   - The target is ray-cast into every frame from the real camera pose. Each pixel keeps the
     nearer of the real room and the target, so occlusion is correct.
   - **Real noise is transplanted:** each frame's real depth error (the frame minus the base map
     rendered from the same pose) is smoothed and added to the target pixels, so the target
     smears exactly like real surfaces did in that frame.
   - Edge "flying pixels" are mixed in, and when the base frames had Hole filling on, the
     silhouette is smoothed the same way.
5. **Map and labels.**
   - The map is integrated with the live mapper's filters (edge, grazing, carving) using the real
     live poses.
   - Boxes come from the target points that actually reached the map (+5 cm, down to the floor):
     snug and exactly aligned.
6. **Output.** Each scene is written like a live scan (`.esd` + `.live.npz` + `.labels.json`), so
   `export` and the Databricks pipeline need no special path.
7. **Always marked.** Every synthetic scan carries:
   - `labels.data_origin = synthetic`, plus `base_scan`, `synth_seed` and `synth_version`
   - `SYNTHETIC` in its notes, and frames with `source = synthetic`

   Silver/gold tables expose `data_origin` (`real` when absent).

## Validation (2026-11-15)
- **Surface thickness:** synthetic body surface noise is 1.3 cm median distance to the true
  shape after the noise transplant (0.6 cm before). Real floor layers in the same scans are
  about 2 cm.
- **Shape:** side views of synthetic lying-on-side and on-back bodies match real scan25/26:
  a 0.25–0.35 m high mound, 1.3–1.7 m long.
- **Known gaps:**
  - No real cloth folds, hair, breathing or motion.
  - No pets with fur.
  - Clutter boxes are clean cuboids.

## Consequences
- **Training:** use synthetic scans for pipeline development and pre-training. **Report accuracy
  on real scans only** (`WHERE data_origin = 'real'`).
- **Real scans still matter.** Real crouching, covered and pet scans should replace synthetic
  ones as they're collected; synthetic scans are a stopgap, not a substitute.
- **Bases must be truly empty** and labelled `empty_room`, or the inserted target will share the
  floor with an unlabelled object.
