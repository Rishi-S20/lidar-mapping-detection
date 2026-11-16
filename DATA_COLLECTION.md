# Collecting labelled LiDAR scans

Goal: scans where people, pets and objects are boxed and labelled, so the next
stage (embeddings → classifier) has training data. Labels are drawn in the live
viewer right after each scan ([ADR-0017](docs/decisions/0017-scan-labelling.md)).

## Setup (once per session)

1. Mac: `cd mapper && .venv/bin/python -m lidar_mapper serve --record recordings/<day>-<room>.esd`
2. Phone: the capture app, **LiDAR only**, the Mac address (see `README.md`; on an
   IPv6-only hotspot use the Mac's IPv6 address in brackets), **Start streaming**.
3. Browser: `http://127.0.0.1:8780/`.

## Per scan (2–3 minutes)

1. **Clear map & start new scan** in the viewer. That starts a new recording file:
   `…-scan2.esd`, `…-scan3.esd`, and so on.
2. Stage the scene (see the table below). Note who or what is where.
3. Scan for 30–60 s: start 1 m+ from walls, sweep slowly, cover the target from 2–3 angles,
   and keep the app streaming (don't tap Stop mid-scan).
4. Fill in the **Label this scan** panel: lights, room, scene, notes.
5. For each person, pet or notable object: choose Label and Pose, click **+ Add object box**,
   then click two opposite corners around it on the floor plan. Height is set from the points
   inside. Make the box snug, with about 10 cm of margin.
6. **Save labels** (it says *saved ✓*). Then go back to step 1.

## What to capture (first target: ~30 scans)

| Scene | Count | Notes |
|---|---|---|
| empty_room | 4 | Same rooms as below, nobody in them (negatives) |
| person_standing | 4 | Different spots and distances (1–4 m) |
| person_sitting | 3 | Chair and floor |
| person_lying | 6 | On the floor, different orientations. **The key case** |
| person_crouching | 3 | |
| person_partly_covered | 4 | Blanket, cardboard, chair over the legs |
| pet | 3 | Real pet or a stuffed animal (say which in notes) |
| mixed / clutter | 3 | Person among boxes, bags and furniture; label the clutter too |

For each scene, do about half with the **lights off** (set Lights = off). That's also
the dark test.

## Export and upload (end of session)

```bash
cd mapper
.venv/bin/python -m lidar_mapper export recordings/<day>-<room>*.esd \
    --upload /Volumes/workspace/dev_<you>_echosense_lidar/landing/lidar
```

Labels and the live map are picked up automatically from `<scan>.labels.json` and
`<scan>.live.npz` next to each recording. The upload triggers the Databricks
pipeline. Check:

- `silver_lidar_annotations`: one row per box
- `gold_lidar_annotation_summary`: box sizes and points inside (0 points means the box missed)
- `gold_lidar_labeled_points`: the labelled point sets

## Tips

- Scans with no boxes are still useful (empty_room negatives); set the scene anyway.
- If the map smears (tracking lost), clear and redo the scan rather than labelling it.
- Keep recordings: the `.esd` files are the raw source of truth and can be re-exported.
