# ADR-0019: Person detector v0 on the LiDAR map (floor-plan cells + random forest)

- **Status:** accepted as a baseline; not good enough to demo as "detects people"
- **Date:** 2026-11-15
- **Area:** perception (LiDAR)

## Context
The plan is: LiDAR maps the room and finds people; the sonar gives the distance to a found person.
Nothing in the repo found people. The surface forest (`src/lidar-model.js`) labels floor, wall,
ceiling and object only. Labelled scans now exist ([ADR-0017](0017-scan-labelling.md),
[ADR-0018](0018-synthetic-lidar-scans.md)).

## Decision
`mapper/src/lidar_mapper/detect.py` (`python -m lidar_mapper.detect eval|train|run`):

1. Find the floor; keep map points 7 cm to 2 m above it. Geometry only, so it works in the dark.
2. Cut the floor plan into 10 cm cells. Each occupied cell gets 30 features: a height profile in
   8 bands, its top height, and the same for the area around it (0.3 to 0.65 m).
3. A random forest (150 trees) scores each cell person / not person. Cells inside a `person` box
   are positives.
4. Neighbouring high-score cells become one detection with a box.

Clustering the map into objects first was tried and dropped: people merged with nearby furniture
in most real scans.

## Results (2026-11-15, real scans only, each real scan held out in turn)
14 real scans, 8 people (1 standing, 3 sitting, 4 lying). A person counts as found when a
detection has half its footprint inside the labelled box.

| Training data | People found | False alarms | Scans with a false alarm |
|---|---|---|---|
| Real + synthetic | 4 of 8 | 5 | 4 of 14 |
| Real only | 4 of 8 | 4 | 3 of 14 |
| Real + synthetic, whole room held out | 3 of 8 | 10 | 6 of 14 |

- Per-cell ranking: AUC 0.88, average precision 0.39 (8% of cells are person cells).
- **Lying people:** 3 of 4 found. In the fourth the person was flagged but so was most of the room.
- **Standing and sitting:** 1 of 4. There is one real standing scan, so holding it out leaves none.
- **Synthetic scans did not help** on real scans in this test.
- The strongest features describe the surroundings, not the body: with one venue, the model
  partly learns the room.
- Dropping the wide-context features found 5 of 8 but doubled false alarms. Not adopted.

## Consequences
- This is a working baseline and an evaluation harness, not a usable detector. Do not quote it
  as detection accuracy.
- The limit is data: 8 real people in 2 spots of one venue. More real scans in more rooms come
  before any model change.
- Real boxes are loose (drawn by hand on the floor plan), so some positive cells are furniture.
- Not done: running it live in the viewer, training on Databricks, the sonar range-to-person step.
