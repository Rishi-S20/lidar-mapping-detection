"""LiDAR lakehouse: Lakeflow Declarative Pipeline (ADR-0016).

Input (landing volume, written by `lidar_mapper export --upload`):
  lidar/frames/<session_id>.parquet   one row per depth frame (raw float16 depth, IMU, mapper pose)
  lidar/maps/<session_id>.parquet     one row per map voxel (x, y, z, hits, segment)
  lidar/sessions/<session_id>.json    session metadata + labels (e.g. lights=off)

Pure Spark/SQL: no custom Python packages on the cluster. Raw depth stays in
bronze so sessions can be reprocessed later; the embedding/classification stage
will read `silver_lidar_map_points` and `silver_lidar_frames`.
"""

import dlt
from pyspark.sql import Window, functions as F

LANDING = spark.conf.get("lidar.landing_path")  # noqa: F821 (provided by the pipeline runtime)

SESSION_SCHEMA = """
  session_id STRING, export_version STRING, protocol_version STRING, mapper_version STRING,
  source_file STRING, exported_at STRING, recorded_at STRING, labels MAP<STRING, STRING>, notes STRING,
  mapper_settings MAP<STRING, STRING>, map_source STRING,
  annotations ARRAY<STRUCT<label STRING, pose STRING, min ARRAY<DOUBLE>, max ARRAY<DOUBLE>, note STRING>>,
  stats STRUCT<frames INT, app_sessions INT, duration_s DOUBLE, status_counts MAP<STRING, INT>,
               tracking_fraction DOUBLE, mean_valid_fraction DOUBLE, map_voxels BIGINT, map_points BIGINT,
               unmerged_segments INT, relocalizations INT, merges INT,
               floor STRUCT<ok BOOLEAN, floor_tilt_deg DOUBLE, floor_z DOUBLE, reason STRING>,
               extent_m ARRAY<DOUBLE>>
"""
POSE_STATUSES = "('waiting', 'init', 'tracking', 'relocalized', 'merged', 'lost', 'arkit', 'not_processed')"


def _autoload(fmt: str, sub: str, **opts):
    r = (spark.readStream.format("cloudFiles")  # noqa: F821
         .option("cloudFiles.format", fmt)
         .option("cloudFiles.schemaEvolutionMode", "none"))
    for k, v in opts.items():
        r = r.option(k, v)
    return r.load(f"{LANDING}/{sub}")


# ---------------------------------------------------------------- bronze

@dlt.table(comment="Raw LiDAR frames as exported: float16 depth bytes, intrinsics, IMU, mapper pose. "
           "Format: PROTOCOL.md + mapper/src/lidar_mapper/export.py.")
def bronze_lidar_frames():
    return (_autoload("parquet", "frames")
            .withColumn("source_file", F.col("_metadata.file_path"))
            .withColumn("ingested_at", F.current_timestamp()))


@dlt.table(comment="Raw session map voxels (seen >= 2 times), per segment, as exported.")
def bronze_lidar_maps():
    return (_autoload("parquet", "maps")
            .withColumn("source_file", F.col("_metadata.file_path"))
            .withColumn("ingested_at", F.current_timestamp()))


@dlt.table(comment="Raw session manifests (one JSON file per session). A re-export adds a new row.")
def bronze_lidar_sessions():
    # allowOverwrites: a re-exported manifest (e.g. fixed labels) has the same file name. Without this,
    # Auto Loader skips it as already ingested and the fix never reaches silver.
    return (_autoload("text", "sessions", wholeText="true",  # camelCase: "wholetext" is silently ignored
                      **{"cloudFiles.allowOverwrites": "true"})
            .withColumn("source_file", F.col("_metadata.file_path"))
            .withColumn("ingested_at", F.current_timestamp()))


# ---------------------------------------------------------------- silver

@dlt.table(comment="Session metadata, flattened, latest export per session only. "
           "`lights` comes from the export label (on/off/unknown).")
@dlt.expect_or_drop("has_session_id", "session_id IS NOT NULL")
@dlt.expect("known_export_version", "export_version LIKE '0.%'")
def silver_lidar_sessions():
    # A materialized view, not a stream: bronze can hold several exports of one session, and only the
    # newest one should count. Session manifests are small, so recomputing this on refresh is cheap.
    s = spark.read.table("bronze_lidar_sessions").select(  # noqa: F821
        F.from_json("value", SESSION_SCHEMA).alias("j"), "source_file", "ingested_at")
    newest_first = Window.partitionBy("j.session_id").orderBy(F.col("j.exported_at").desc(), F.col("ingested_at").desc())
    s = s.withColumn("_rank", F.row_number().over(newest_first)).where("_rank = 1").drop("_rank")
    return s.select(
        "j.session_id", "j.export_version", "j.source_file", F.to_timestamp("j.recorded_at").alias("recorded_at"),
        "j.labels", F.coalesce(F.col("j.labels")["lights"], F.lit("unknown")).alias("lights"),
        F.col("j.labels")["room"].alias("room"), "j.notes", "j.mapper_version", "j.mapper_settings",
        F.col("j.stats.frames").alias("frames"), F.col("j.stats.duration_s").alias("duration_s"),
        F.col("j.stats.app_sessions").alias("app_sessions"),
        F.col("j.stats.tracking_fraction").alias("tracking_fraction"),
        F.col("j.stats.mean_valid_fraction").alias("mean_valid_fraction"),
        F.col("j.stats.map_points").alias("map_points"), F.col("j.stats.unmerged_segments").alias("unmerged_segments"),
        F.col("j.stats.floor.floor_tilt_deg").alias("floor_tilt_deg"), F.col("j.stats.extent_m").alias("extent_m"),
        F.col("j.labels")["scene"].alias("scene"), "j.map_source",
        # 'synthetic' = target inserted into a real empty-room scan (ADR-0018); filter on it for evaluation
        F.coalesce(F.col("j.labels")["data_origin"], F.lit("real")).alias("data_origin"),
        F.greatest(F.size("j.annotations"), F.lit(0)).alias("n_annotations"), "j.annotations",
        "ingested_at")


@dlt.table(comment="Frames that pass integrity checks, with derived columns. Depth stays as raw bytes.")
@dlt.expect_or_drop("depth_size_matches", "length(depth_f16) = width * height * 2")
@dlt.expect_or_drop("has_intrinsics", "fx > 0 AND fy > 0 AND cx > 0 AND cy > 0")
@dlt.expect_or_drop("pose_is_3x3_plus_t", "size(pose_R) = 9 AND size(pose_t) = 3")
@dlt.expect("known_pose_status", f"pose_status IN {POSE_STATUSES}")
@dlt.expect("valid_fraction_in_range", "valid_fraction BETWEEN 0 AND 1")
@dlt.expect("gravity_measured", "size(gravity) = 3 AND aggregate(gravity, 0D, (a, x) -> a + x * x) > 0.5")
def silver_lidar_frames():
    f = spark.readStream.table("bronze_lidar_frames")  # noqa: F821
    return (f.withColumn("is_tracked", F.col("pose_status").isin("tracking", "relocalized", "merged", "init"))
             .withColumn("pos_x", F.col("pose_t")[0]).withColumn("pos_y", F.col("pose_t")[1])
             .withColumn("pos_z", F.col("pose_t")[2])
             .drop("source_file"))


@dlt.table(comment="Map points per session and segment (z up, metres, segment 0.. in its own frame). "
           "Input for segmentation / embeddings.")
@dlt.expect_or_drop("finite_xyz", "NOT (isnan(x) OR isnan(y) OR isnan(z))")
@dlt.expect_or_drop("seen_twice", "hits >= 2")
@dlt.expect("plausible_extent", "abs(x) < 50 AND abs(y) < 50 AND abs(z) < 10")
def silver_lidar_map_points():
    return spark.readStream.table("bronze_lidar_maps").drop("source_file")  # noqa: F821


# ---------------------------------------------------------------- gold

@dlt.table(comment="One row per session: capture quality, tracking, and map size.")
def gold_lidar_session_summary():
    fr = (spark.read.table("silver_lidar_frames")  # noqa: F821
          .groupBy("session_id")
          .agg(F.count("*").alias("frames_ingested"),
               F.avg(F.col("is_tracked").cast("double")).alias("tracked_fraction"),
               F.avg((F.col("pose_status") == "lost").cast("double")).alias("lost_fraction"),
               F.avg("valid_fraction").alias("mean_valid_depth"),
               F.expr("percentile_approx(icp_rmse, 0.5)").alias("median_icp_rmse_m"),
               (F.max("t") - F.min("t")).alias("duration_s"),
               F.max("app_session").alias("app_restarts")))
    mp = (spark.read.table("silver_lidar_map_points")  # noqa: F821
          .groupBy("session_id")
          .agg(F.count("*").alias("map_points"), F.countDistinct("segment").alias("segments"),
               (F.max("x") - F.min("x")).alias("extent_x_m"), (F.max("y") - F.min("y")).alias("extent_y_m"),
               (F.max("z") - F.min("z")).alias("extent_z_m")))
    se = spark.read.table("silver_lidar_sessions").select(  # noqa: F821
        "session_id", "recorded_at", "lights", "room", "notes", "floor_tilt_deg")
    return se.join(fr, "session_id", "left").join(mp, "session_id", "left")


@dlt.table(comment="Capture quality by lighting label: the lights-on vs lights-off test (ADR-0015).")
def gold_lidar_quality_by_lights():
    f = spark.read.table("silver_lidar_frames")  # noqa: F821
    s = spark.read.table("silver_lidar_sessions").select("session_id", "lights")  # noqa: F821
    return (f.join(s, "session_id", "left").fillna({"lights": "unknown"})
             .groupBy("lights")
             .agg(F.countDistinct("session_id").alias("sessions"), F.count("*").alias("frames"),
                  F.avg("valid_fraction").alias("mean_valid_depth"),
                  F.avg(F.col("is_tracked").cast("double")).alias("tracked_fraction"),
                  F.expr("percentile_approx(icp_rmse, 0.5)").alias("median_icp_rmse_m")))


@dlt.table(comment="Per session, per 5-second window: depth validity and tracking (for charts).")
def gold_lidar_timeline():
    f = spark.read.table("silver_lidar_frames")  # noqa: F821
    t0 = f.groupBy("session_id").agg(F.min("t").alias("t0"))
    return (f.join(t0, "session_id")
             .withColumn("window_s", (F.floor((F.col("t") - F.col("t0")) / 5) * 5).cast("int"))
             .groupBy("session_id", "window_s")
             .agg(F.count("*").alias("frames"), F.avg("valid_fraction").alias("mean_valid_depth"),
                  F.avg(F.col("is_tracked").cast("double")).alias("tracked_fraction"),
                  F.avg("pos_x").alias("pos_x"), F.avg("pos_y").alias("pos_y")))


# ---------------------------------------------------------------- labels (ADR-0017)

@dlt.table(comment="Object boxes drawn in the live viewer, one row per box (map frame of that session, metres).")
@dlt.expect_or_drop("box_has_extent", "x_max > x_min AND y_max > y_min AND z_max > z_min")
@dlt.expect("known_label", "label IN ('person', 'pet', 'furniture', 'clutter', 'door', 'other')")
def silver_lidar_annotations():
    s = spark.read.table("silver_lidar_sessions").where("n_annotations > 0")  # noqa: F821
    b = s.select("session_id", "lights", "room", "scene", "data_origin",
                 F.posexplode("annotations").alias("box_id", "a"))
    return b.select(
        "session_id", "box_id", F.col("a.label").alias("label"), F.col("a.pose").alias("pose"),
        F.col("a.note").alias("note"), "lights", "room", "scene", "data_origin",
        F.col("a.min")[0].alias("x_min"), F.col("a.min")[1].alias("y_min"), F.col("a.min")[2].alias("z_min"),
        F.col("a.max")[0].alias("x_max"), F.col("a.max")[1].alias("y_max"), F.col("a.max")[2].alias("z_max"))


@dlt.table(comment="Map points inside each labelled box: the labelled point sets for embeddings/classification.")
def gold_lidar_labeled_points():
    pts = spark.read.table("silver_lidar_map_points").alias("p")  # noqa: F821
    box = spark.read.table("silver_lidar_annotations").alias("b")  # noqa: F821
    cond = ((F.col("p.session_id") == F.col("b.session_id"))
            & F.col("p.x").between(F.col("b.x_min"), F.col("b.x_max"))
            & F.col("p.y").between(F.col("b.y_min"), F.col("b.y_max"))
            & F.col("p.z").between(F.col("b.z_min"), F.col("b.z_max")))
    return pts.join(box, cond).select(
        "p.session_id", "b.box_id", "b.label", "b.pose", "b.lights", "b.scene", "b.data_origin",
        "p.x", "p.y", "p.z", "p.hits")


@dlt.table(comment="One row per labelled box: size and how many map points support it.")
def gold_lidar_annotation_summary():
    lp = spark.read.table("gold_lidar_labeled_points")  # noqa: F821
    counts = lp.groupBy("session_id", "box_id").agg(F.count("*").alias("points_inside"))
    return (spark.read.table("silver_lidar_annotations")  # noqa: F821
            .join(counts, ["session_id", "box_id"], "left").fillna({"points_inside": 0})
            .withColumn("size_x_m", F.col("x_max") - F.col("x_min"))
            .withColumn("size_y_m", F.col("y_max") - F.col("y_min"))
            .withColumn("height_m", F.col("z_max") - F.col("z_min")))
