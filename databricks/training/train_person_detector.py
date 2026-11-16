"""Train and evaluate the LiDAR person detector on Databricks (ADR-0019).

Reads the lakehouse tables (silver_lidar_sessions, silver_lidar_map_points, silver_lidar_annotations),
builds the same floor-plan features as `lidar_mapper.detect`, then:

  1. Evaluates on REAL scans only, holding each real scan out of training in turn.
  2. Trains the final model on every labelled scan (real + synthetic).
  3. Logs parameters, metrics and the model to MLflow, and registers it in Unity Catalog.
  4. Writes the per-scan results to `gold_person_detector_eval`.

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
            .select("session_id", "source_file", "scene", "room", "lights", "data_origin",
                    F.col("labels")["base_scan"].alias("base_scan"))
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
                        row.data_origin or "real", row.room or "", Path(row.base_scan or "").stem))
    scans[-1].scene, scans[-1].session_id, scans[-1].lights = row.scene, row.session_id, row.lights

real = [s for s in scans if s.origin == "real"]
synth = [s for s in scans if s.origin != "real"]
print(f"{len(real)} real scans ({sum(len(s.boxes) for s in real)} people), "
      f"{len(synth)} synthetic ({sum(len(s.boxes) for s in synth)} people)")

by_name = {s.name: s for s in scans}
results = {}
for label, use_synth in (("real_plus_synthetic", True), ("real_only", False)):
    rows = D.evaluate(scans, "scan", use_synth, a.threshold)
    results[label] = (rows, D.summarize(rows))
    print(label, results[label][1])

model = D.fit(scans)
with mlflow.start_run(run_name="person-detector-v0") as run:
    mlflow.log_params({"cell_m": D.CELL, "features": len(D.FEATURES), "trees": model.n_estimators,
                       "threshold": a.threshold, "real_scans": len(real), "synthetic_scans": len(synth),
                       "real_people": sum(len(s.boxes) for s in real),
                       "evaluation": "each real scan held out in turn; synthetic scans built on it held out too"})
    for label, (rows, summary) in results.items():
        mlflow.log_metrics({f"{label}_people_found": summary["found"], f"{label}_people": summary["people"],
                            f"{label}_recall": summary["found"] / max(summary["people"], 1),
                            f"{label}_false_alarms": summary["false_alarms"],
                            f"{label}_scans_with_false_alarm": summary["scans_with_false_alarm"]})
    for scene in sorted({s.scene for s in real if s.boxes}):
        rows = [r for r in results["real_plus_synthetic"][0] if by_name[r["scan"]].scene == scene]
        mlflow.log_metric(f"found_{scene}", sum(r["found"] for r in rows))
        mlflow.log_metric(f"people_{scene}", sum(r["boxes"] for r in rows))
    X = scans[0].grid.features[:50]
    info = mlflow.sklearn.log_model(model, "model", signature=infer_signature(X, model.predict_proba(X)[:, 1]),
                                    input_example=X[:2])
    try:
        version = mlflow.register_model(info.model_uri, f"{ns}.person_detector")
        print(f"registered {ns}.person_detector version {version.version}")
    except Exception as exc:  # the run and metrics are still logged
        print(f"model not registered: {exc}")
    run_id = run.info.run_id

now = datetime.now(timezone.utc)
table = [(run_id, now, label, r["scan"], by_name[r["scan"]].session_id, by_name[r["scan"]].scene, r["room"],
          by_name[r["scan"]].lights, int(r["boxes"]), int(r["found"]), int(r["false_alarms"]))
         for label, (rows, _) in results.items() for r in rows]
(spark.createDataFrame(table, "run_id string, evaluated_at timestamp, training_data string, scan string, "
                              "session_id string, scene string, room string, lights string, people int, "
                              "found int, false_alarms int")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{ns}.gold_person_detector_eval"))
print(f"wrote {ns}.gold_person_detector_eval ({len(table)} rows), MLflow run {run_id}")
