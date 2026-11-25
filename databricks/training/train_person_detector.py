"""Train and evaluate the LiDAR person detector on Databricks (ADR-0019).

Reads the lakehouse tables (silver_lidar_sessions, silver_lidar_map_points, silver_lidar_annotations),
builds the same floor-plan features as `lidar_mapper.detect`, then:

  1. Evaluates by holding out one whole room at a time: train on the other rooms, test on that one.
     A model scored on a scan of a room it trained on partly scores how well it learned the room.
  2. Trains the final model on every labelled scan.
  3. Logs parameters, metrics and the model to MLflow.
  4. Writes the per-scan results to `gold_person_detector_eval`.
  5. Registers the model in Unity Catalog. If that fails, the job fails (after steps 3 and 4).

Only real scans are used. Old synthetic sessions (ADR-0018, generator removed) may still be in the
tables, so they are filtered out by `data_origin`.

Run: databricks bundle run person_detector_training   (from the repo root)
"""
import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession, functions as F

ap = argparse.ArgumentParser()
ap.add_argument("--catalog", required=True)
ap.add_argument("--schema", required=True)
ap.add_argument("--detect-path", required=True, help="workspace path of lidar_mapper/detect.py")
ap.add_argument("--threshold", type=float, default=0.5)
a = ap.parse_args()

# detect.py is plain numpy/scipy/scikit-learn, so it is loaded straight from the synced file.
spec = importlib.util.spec_from_file_location("detect", a.detect_path)
D = importlib.util.module_from_spec(spec)
sys.modules["detect"] = D
spec.loader.exec_module(D)

spark = SparkSession.builder.getOrCreate()
ns = f"{a.catalog}.{a.schema}"

# MLflow is set up first so a tracking problem fails in seconds, not after the evaluation.
# The registry URI must be set before anything else: on serverless, MLflow otherwise tries to read
# a Spark setting (spark.mlflow.modelRegistryUri) that serverless does not expose.
import os  # noqa: E402

os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/echosense-person-detector")
print("MLflow experiment ready")

sessions = (spark.table(f"{ns}.silver_lidar_sessions")
            .where("scene IS NOT NULL AND scene <> ''")       # never labelled: can't be called a negative
            .where("data_origin = 'real'")                    # leftover synthetic sessions stay out
            .select("session_id", "source_file", "scene", "room", "lights")
            .toPandas())
boxes = spark.table(f"{ns}.silver_lidar_annotations").where("label = 'person'").toPandas()

scans = []
for row in sessions.itertuples():
    pts = (spark.table(f"{ns}.silver_lidar_map_points").where(F.col("session_id") == row.session_id)
           .select("x", "y", "z", "hits").toPandas())
    if len(pts) < 5000:
        print(f"skip {row.session_id}: only {len(pts)} map points")
        continue
    grid = D.build_grid(pts[["x", "y", "z"]].to_numpy(np.float64), pts["hits"].to_numpy())
    own = [{"label": "person", "pose": b.pose, "min": [b.x_min, b.y_min, b.z_min], "max": [b.x_max, b.y_max, b.z_max]}
           for b in boxes[boxes.session_id == row.session_id].itertuples()]
    scans.append(D.Scan(Path(row.source_file or row.session_id).stem, grid, D.cell_labels(grid, own), own,
                        "real", row.room or "", ""))
    scans[-1].scene, scans[-1].session_id, scans[-1].lights = row.scene, row.session_id, row.lights

rooms = sorted({s.room for s in scans})
print(f"{len(scans)} scans, {sum(len(s.boxes) for s in scans)} people, {len(rooms)} rooms: {rooms}")
if "" in rooms:
    print("note: scans with no room label are held out together as one room")

by_name = {s.name: s for s in scans}
rows = D.evaluate(scans, "room", use_synthetic=False, threshold=a.threshold)
summary = D.summarize(rows)
print("hold out one room:", summary)

model = D.fit(scans)
with mlflow.start_run(run_name="person-detector-v0") as run:
    mlflow.log_params({"cell_m": D.CELL, "features": len(D.FEATURES), "trees": model.n_estimators,
                       "threshold": a.threshold, "scans": len(scans), "rooms": len(rooms),
                       "people": sum(len(s.boxes) for s in scans),
                       "evaluation": "each room held out in turn"})
    mlflow.log_metrics({"people_found": summary["found"], "people": summary["people"],
                        "recall": summary["found"] / max(summary["people"], 1),
                        "false_alarms": summary["false_alarms"],
                        "scans_with_false_alarm": summary["scans_with_false_alarm"]})
    for scene in sorted({s.scene for s in scans if s.boxes}):
        scene_rows = [r for r in rows if by_name[r["scan"]].scene == scene]
        mlflow.log_metric(f"found_{scene}", sum(r["found"] for r in scene_rows))
        mlflow.log_metric(f"people_{scene}", sum(r["boxes"] for r in scene_rows))
    X = scans[0].grid.features[:50]
    info = mlflow.sklearn.log_model(model, "model", signature=infer_signature(X, model.predict_proba(X)[:, 1]),
                                    input_example=X[:2])
    run_id = run.info.run_id

now = datetime.now(timezone.utc)
table = [(run_id, now, r["scan"], by_name[r["scan"]].session_id, by_name[r["scan"]].scene, r["room"],
          by_name[r["scan"]].lights, int(r["boxes"]), int(r["found"]), int(r["false_alarms"]))
         for r in rows]
(spark.createDataFrame(table, "run_id string, evaluated_at timestamp, scan string, session_id string, "
                              "scene string, room string, lights string, people int, found int, false_alarms int")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{ns}.gold_person_detector_eval"))
print(f"wrote {ns}.gold_person_detector_eval ({len(table)} rows), MLflow run {run_id}")

# Last, and not caught: a registration failure should turn the job red, not hide in the log.
version = mlflow.register_model(info.model_uri, f"{ns}.person_detector")
print(f"registered {ns}.person_detector version {version.version}")
