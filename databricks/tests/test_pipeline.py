"""Local checks for databricks/pipeline/lidar_pipeline.py, with plain PySpark and a stand-in `dlt`.

Run from the repo root (needs Java and `pip install pyspark pytest`):
    python -m pytest databricks/tests

Only the table functions' queries are tested here. Auto Loader, expectations and the pipeline
runtime itself are Databricks-only, so the decorators are replaced with no-ops.
"""
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline" / "lidar_pipeline.py"


@pytest.fixture(scope="module")
def pipe():
    spark = (SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "1").getOrCreate())
    spark.conf.set("lidar.landing_path", "/unused")
    dlt = types.ModuleType("dlt")
    dlt.table = dlt.expect = dlt.expect_or_drop = lambda *a, **k: (lambda f: f)
    sys.modules["dlt"] = dlt
    ns = {"spark": spark, "__name__": "lidar_pipeline"}
    exec(compile(PIPELINE.read_text(), str(PIPELINE), "exec"), ns)
    yield spark, ns
    spark.catalog.dropTempView("bronze_lidar_sessions")
    spark.catalog.dropTempView("silver_lidar_sessions")


def manifest(sid: str, exported_at: str, box_widths: list[float]) -> str:
    return json.dumps({
        "session_id": sid, "export_version": "0.1", "exported_at": exported_at,
        "recorded_at": "2026-11-15T14:00:00+00:00", "labels": {"scene": "person_lying", "room": "lab"},
        "annotations": [{"label": "person", "pose": "lying", "min": [0, 0, 0], "max": [1, w, 1], "note": ""}
                        for w in box_widths],
        "stats": {"frames": 10}})


def test_label_fix_replaces_old_export(pipe):
    """Scan A was exported twice (labels fixed the second time): only the newest export counts."""
    spark, ns = pipe
    rows = [
        (manifest("A", "2026-11-15T15:00:00+00:00", [0.5]), "a.json", "2026-11-15 15:01:00"),
        (manifest("A", "2026-11-16T09:00:00+00:00", [0.6, 0.8]), "a.json", "2026-11-16 09:01:00"),
        (manifest("B", "2026-11-15T15:05:00+00:00", [0.7]), "b.json", "2026-11-15 15:06:00"),
    ]
    (spark.createDataFrame(rows, "value string, source_file string, ingested_at string")
     .selectExpr("value", "source_file", "cast(ingested_at as timestamp) ingested_at")
     .createOrReplaceTempView("bronze_lidar_sessions"))

    sessions = ns["silver_lidar_sessions"]()
    sessions.createOrReplaceTempView("silver_lidar_sessions")
    assert sorted((r.session_id, r.n_annotations) for r in sessions.collect()) == [("A", 2), ("B", 1)]

    boxes = ns["silver_lidar_annotations"]().collect()
    assert sorted((r.session_id, r.y_max) for r in boxes) == [("A", 0.6), ("A", 0.8), ("B", 0.7)]
