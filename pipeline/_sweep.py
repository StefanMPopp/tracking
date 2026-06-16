"""
Threshold sweep: run TGrabs + TRex for each candidate detect_threshold,
annotate frames with per-blob metrics, save one clip per threshold, and
compose a side-by-side grid video.

All outputs for one sweep run go into the sweep_dir passed by tune.py:
    tuning/sweep_{timestamp}/
        run_threshold_{value}.settings   (one per threshold)
        pv/
        csv/
        annotated_threshold_{value}.mp4  (one per threshold)
        grid_comparison.mp4
"""

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _settings import patch_settings

logger = logging.getLogger(__name__)

# Annotation colours (BGR)
COLOUR_BOX  = (50, 220, 50)    # green bounding box
COLOUR_TEXT = (255, 255, 255)  # white labels
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.35
FONT_THICKNESS = 1


# =============================================================================
# Public entry point
# =============================================================================

def run_sweep(
    trial: str,
    project_dir: Path,
    sweep_dir: Path,
    base_settings_file: Path,
    pipeline_config: dict,
    effective_config: dict,
) -> Path:
    """
    Run TGrabs + TRex for each threshold in effective_config["sweep_thresholds"],
    render one annotated clip per threshold, compose a grid video.

    sweep_dir is created by tune.py before calling this function.
    Returns the path to grid_comparison.mp4.
    """
    thresholds             = effective_config["sweep_thresholds"]
    video_conversion_range = effective_config["video_conversion_range"]
    meta_real_width        = effective_config["meta_real_width"]
    track_max_individuals  = effective_config["track_max_individuals"]
    individual_prefix      = effective_config["individual_prefix"]

    videos_dir = project_dir / "1_videos"
    video_file = videos_dir / f"{trial}.MP4"
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    pv_dir  = sweep_dir / "pv"
    csv_dir = sweep_dir / "csv"
    pv_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Copy background image into sweep pv/ so TGrabs picks it up
    from _background import copy_background_to_sweep
    copy_background_to_sweep(
        trial=trial,
        project_pv_dir=project_dir / "2_pv",
        sweep_pv_dir=pv_dir,
    )

    # --------------------------------------------------------------------------
    clip_files = []

    for threshold in thresholds:
        logger.info("=" * 50)
        logger.info("Threshold %d", threshold)
        logger.info("=" * 50)

        settings_file = sweep_dir / f"run_threshold_{threshold:03d}.settings"
        patch_settings(
            base_settings_file=base_settings_file,
            output_file=settings_file,
            overrides={
                "detect_threshold":       threshold,
                "meta_real_width":        meta_real_width,
                "track_max_individuals":  track_max_individuals,
                "individual_prefix":      individual_prefix,
                "video_conversion_range": video_conversion_range,
            },
        )

        _run_tgrabs(
            video_file=video_file,
            settings_file=settings_file,
            trial=trial,
            pv_dir=pv_dir,
            pipeline_config=pipeline_config,
            video_conversion_range=video_conversion_range,
            detect_threshold=threshold,
        )

        pv_file = pv_dir / f"{trial}.pv"
        if not pv_file.exists():
            logger.warning(
                "TGrabs did not produce a .pv file for threshold=%d — skipping.", threshold
            )
            continue

        _run_trex(
            pv_file=pv_file,
            settings_file=settings_file,
            trial=trial,
            csv_dir=csv_dir,
            pipeline_config=pipeline_config,
        )

        clip_file = sweep_dir / f"annotated_threshold_{threshold:03d}.mp4"
        _render_annotated_clip(
            trial=trial,
            csv_dir=csv_dir,
            video_file=video_file,
            video_conversion_range=video_conversion_range,
            meta_real_width=meta_real_width,
            threshold=threshold,
            output_file=clip_file,
        )
        clip_files.append((threshold, clip_file))

    # --------------------------------------------------------------------------
    grid_video_file = sweep_dir / "grid_comparison.mp4"
    if clip_files:
        _compose_grid_video(
            clip_files=[clip_file for _, clip_file in clip_files],
            thresholds=[threshold for threshold, _ in clip_files],
            grid_cols=effective_config.get("grid_cols", 2),
            output_file=grid_video_file,
        )
    else:
        logger.error("No clips were produced — cannot compose grid video.")

    return grid_video_file


# =============================================================================
# TRex invocation
# =============================================================================

def _conda_prefix(pipeline_config: dict) -> str:
    miniforge_dir = pipeline_config.get("miniforge_dir", "~/miniforge3")
    conda_env     = pipeline_config.get("conda_env", "trex")
    return (
        f"source {miniforge_dir}/etc/profile.d/conda.sh && "
        f"conda activate {conda_env}"
    )


def _run_tgrabs(
    video_file: Path,
    settings_file: Path,
    trial: str,
    pv_dir: Path,
    pipeline_config: dict,
    video_conversion_range: list,
    detect_threshold: int,
) -> None:
    range_str = f"[{video_conversion_range[0]},{video_conversion_range[1]}]"
    command = (
        f"{_conda_prefix(pipeline_config)} && "
        f"trex -task convert "
        f'-i "{video_file}" '
        f'-s "{settings_file}" '
        f'-video_conversion_range "{range_str}" '
        f'-detect_threshold "{detect_threshold}" '
        f'-o "{trial}" '
        f'-d "{pv_dir}"'
    )
    logger.info("TGrabs: %s", command)
    subprocess.run(["bash", "-c", command], check=True)


def _run_trex(
    pv_file: Path,
    settings_file: Path,
    trial: str,
    csv_dir: Path,
    pipeline_config: dict,
) -> None:
    command = (
        f"{_conda_prefix(pipeline_config)} && "
        f"trex -task track "
        f'-i "{pv_file}" '
        f'-s "{settings_file}" '
        f'-data_prefix "{trial}" '
        f'-output_dir "{csv_dir}"'
    )
    logger.info("TRex: %s", command)
    subprocess.run(["bash", "-c", command], check=True)


# =============================================================================
# Frame annotation
# =============================================================================

def _render_annotated_clip(
    trial: str,
    csv_dir: Path,
    video_file: Path,
    video_conversion_range: list,
    meta_real_width: float,
    threshold: int,
    output_file: Path,
) -> None:
    """
    Render an annotated clip for one threshold.

    Each tracked blob is drawn with a bounding box and labelled:
      - individual ID (short)
      - blob area in pixels and cm²
      - frame-to-frame centroid speed in cm/s
    """
    start_frame, end_frame = video_conversion_range

    csv_files = sorted(csv_dir.glob(f"{trial}_*.csv"))
    if not csv_files:
        logger.warning("No CSV files in %s — skipping annotation.", csv_dir)
        return

    trajectories_df = _load_trajectories(csv_files)

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")

    frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = capture.get(cv2.CAP_PROP_FPS) or 25.0
    cm_per_pixel = meta_real_width / frame_width

    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    speed_lookup = _compute_speeds(trajectories_df, fps, cm_per_pixel)

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_index in range(start_frame, end_frame):
        success, frame = capture.read()
        if not success:
            break

        frame = _annotate_frame(
            frame=frame,
            frame_index=frame_index,
            trajectories_df=trajectories_df,
            speed_lookup=speed_lookup,
            cm_per_pixel=cm_per_pixel,
        )

        # Header bar
        cv2.rectangle(frame, (0, 0), (frame_width, 26), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"threshold={threshold}  frame={frame_index}",
            (6, 18), FONT, 0.5, COLOUR_TEXT, 1, cv2.LINE_AA,
        )

        writer.write(frame)

    capture.release()
    writer.release()
    logger.info("Annotated clip: %s", output_file)


def _load_trajectories(csv_files: list[Path]) -> pd.DataFrame:
    """Load and concatenate per-individual CSVs; normalise column names."""
    individual_dfs = []
    for csv_file in csv_files:
        individual_df = pd.read_csv(csv_file)
        individual_df.columns = [
            col.strip().lstrip("#").strip().lower()
            for col in individual_df.columns
        ]
        individual_df["individual_id"] = csv_file.stem
        individual_dfs.append(individual_df)
    return pd.concat(individual_dfs, ignore_index=True)


def _compute_speeds(
    trajectories_df: pd.DataFrame,
    fps: float,
    cm_per_pixel: float,
) -> dict:
    """
    Compute frame-to-frame centroid speed (cm/s) per individual.

    Returns: {individual_id: {frame_index: speed_cm_per_s}}
    """
    speed_lookup = {}
    for individual_id, individual_df in trajectories_df.groupby("individual_id"):
        individual_df = individual_df.sort_values("frame")
        speeds        = {}
        previous_row  = None
        for _, row in individual_df.iterrows():
            frame_index = int(row["frame"])
            if previous_row is not None:
                dx_pixels      = row["x"] - previous_row["x"]
                dy_pixels      = row["y"] - previous_row["y"]
                distance_cm    = np.hypot(dx_pixels, dy_pixels) * cm_per_pixel
                speed_cm_per_s = distance_cm * fps
                speeds[frame_index] = speed_cm_per_s
            previous_row = row
        speed_lookup[individual_id] = speeds
    return speed_lookup


def _annotate_frame(
    frame: np.ndarray,
    frame_index: int,
    trajectories_df: pd.DataFrame,
    speed_lookup: dict,
    cm_per_pixel: float,
) -> np.ndarray:
    """Draw bounding boxes and metric labels for all blobs in one frame."""
    frame_df = trajectories_df[trajectories_df["frame"] == frame_index]

    for _, row in frame_df.iterrows():
        if pd.isna(row.get("x")) or pd.isna(row.get("y")):
            continue

        cx = int(row["x"])
        cy = int(row["y"])
        blob_width_px  = int(row.get("blob_width",  20))
        blob_height_px = int(row.get("blob_height", 20))

        x1 = cx - blob_width_px  // 2
        y1 = cy - blob_height_px // 2
        x2 = cx + blob_width_px  // 2
        y2 = cy + blob_height_px // 2

        num_pixels = row.get("num_pixels", blob_width_px * blob_height_px)
        area_cm2   = num_pixels * (cm_per_pixel ** 2)

        individual_id  = row["individual_id"]
        speed_cm_per_s = speed_lookup.get(individual_id, {}).get(frame_index)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOUR_BOX, 1)

        short_id    = str(individual_id).split("_")[-1]
        label_area  = f"{int(num_pixels)}px {area_cm2:.3f}cm2"
        label_speed = (
            f"spd:{speed_cm_per_s:.1f}cm/s"
            if speed_cm_per_s is not None
            else "spd:--"
        )

        cv2.putText(frame, short_id,    (x1, y1 - 30), FONT, FONT_SCALE, COLOUR_TEXT, FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, label_area,  (x1, y1 - 18), FONT, FONT_SCALE, COLOUR_TEXT, FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, label_speed, (x1, y1 -  6), FONT, FONT_SCALE, COLOUR_TEXT, FONT_THICKNESS, cv2.LINE_AA)

    return frame


# =============================================================================
# Grid compositor
# =============================================================================

def _compose_grid_video(
    clip_files: list[Path],
    thresholds: list[int],
    grid_cols: int,
    output_file: Path,
) -> None:
    """Tile annotated clips into a single side-by-side grid video."""
    if not clip_files:
        return

    sample_capture = cv2.VideoCapture(str(clip_files[0]))
    clip_width  = int(sample_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    clip_height = int(sample_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = sample_capture.get(cv2.CAP_PROP_FPS) or 25.0
    sample_capture.release()

    grid_rows   = (len(clip_files) + grid_cols - 1) // grid_cols
    grid_width  = clip_width  * grid_cols
    grid_height = clip_height * grid_rows

    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (grid_width, grid_height),
    )

    captures    = [cv2.VideoCapture(str(clip_file)) for clip_file in clip_files]
    blank_frame = np.zeros((clip_height, clip_width, 3), dtype=np.uint8)

    while True:
        frames         = []
        any_frame_read = False
        for capture in captures:
            success, frame = capture.read()
            if success:
                any_frame_read = True
                frames.append(frame)
            else:
                frames.append(blank_frame.copy())

        if not any_frame_read:
            break

        while len(frames) < grid_rows * grid_cols:
            frames.append(blank_frame.copy())

        rows = [
            np.hstack(frames[row_index * grid_cols: (row_index + 1) * grid_cols])
            for row_index in range(grid_rows)
        ]
        writer.write(np.vstack(rows))

    for capture in captures:
        capture.release()
    writer.release()
    logger.info("Grid video: %s", output_file)
