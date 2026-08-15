"""
Threshold sweep: run TGrabs + TRex for each candidate detect_threshold,
annotate frames with per-blob metrics, and save one clip per threshold.

All outputs go directly into tuning_dir (passed by tune.py):
    tuning/
        tuning_log.json
        2_pv/
            {video_name}.pv               ← overwritten each TGrabs run
            average_{video_name}.png      ← background image (if present)
        {video_name}_thresh{value}.settings   (one per threshold)
        csv_{video_name}_t{value}/        (one per threshold)
            {video_name}_0.csv
            {video_name}_1.csv
            ...
        {video_name}_t{value}.mp4         (one annotated clip per threshold)

TRex byproducts cleaned up after each track run:
    2_pv/{video_name}.results
    2_pv/{video_name}.settings
    2_pv/{video_name}.settings.backup
    csv_{video_name}_t{value}/{video_name}.results.meta
"""

import logging
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ._parameters import collect_overrides
from ._settings import patch_settings

logger = logging.getLogger(__name__)

# Cooperative cancellation for a running sweep. A sweep checks this event
# between thresholds (and the subprocess helpers register the currently
# running Popen handle so it can be terminated immediately rather than
# waiting for the whole sweep loop to notice). Set via request_sweep_stop(),
# called from both the Tune tab's Stop button and a SIGINT handler.
_stop_event = threading.Event()
_current_process: subprocess.Popen | None = None
_current_process_lock = threading.Lock()


def request_sweep_stop() -> None:
    """
    Request cancellation of any currently running sweep. Sets the stop
    flag (checked between thresholds) and terminates the in-flight
    TGrabs/TRex subprocess, if any, so a single long threshold doesn't
    have to finish before the sweep actually stops.
    """
    _stop_event.set()
    with _current_process_lock:
        if _current_process is not None and _current_process.poll() is None:
            logger.info("Terminating in-flight subprocess (pid=%d)…", _current_process.pid)
            _current_process.terminate()


def _clear_sweep_stop() -> None:
    """Reset the stop flag at the start of a new sweep."""
    _stop_event.clear()

# Annotation colours (BGR)
# INDIVIDUAL_COLOURS cycles over individual IDs; chosen to be visually distinct
# and legible on both light and dark arena backgrounds.
INDIVIDUAL_COLOURS = [
    ( 50, 220,  50),   # green
    ( 50,  50, 220),   # red
    (220, 180,  50),   # cyan-blue
    (220,  50, 220),   # magenta
    ( 50, 220, 220),   # yellow
    (255, 140,   0),   # blue (bright)
    (  0, 165, 255),   # orange
    (180,  50, 220),   # purple-red
]
COLOUR_TEXT    = (255, 255, 255)
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.35
FONT_THICKNESS = 1


# =============================================================================
# Public entry point
# =============================================================================

def run_sweep(
    video_name: str,
    project_dir: Path,
    videos_dir: Path,
    masks_dir: Path,
    tuning_dir: Path,
    base_settings_file: Path,
    pipeline_config: dict,
    effective_config: dict,
    project_config: dict,
) -> list[Path]:
    """
    Run TGrabs + TRex for each threshold in effective_config["sweep_thresholds"]
    and render one annotated clip per threshold into tuning_dir.

    videos_dir is the folder containing the source video — pass project_dir /
    "1_videos" for a real trial video, or the tuning-mode videos folder
    (e.g. project_dir / "2_tracking" / "tuning" / "1_videos") for a tuning clip.

    masks_dir is resolved the same way: project-level masks/ for real videos,
    or tuning/masks/ for dedicated tuning clips (which are not part of any
    batch and are not expected to share masks with real trial videos).

    project_config (the raw, unresolved project.yaml contents) is needed
    separately from effective_config to resolve which batch (if any) this
    video belongs to, for mask resolution.

    Returns a list of produced clip file paths.
    """
    _clear_sweep_stop()

    thresholds             = effective_config["sweep_thresholds"]
    video_conversion_range = effective_config.get("video_conversion_range")  # None if not set
    animal_size_min        = effective_config.get("animal_size_min")
    animal_size_max        = effective_config.get("animal_size_max")
    if video_conversion_range is None:
        # Fall back to the value in default.settings so the annotated clip
        # uses the same range that TGrabs will use.
        from ._settings import read_settings
        default_settings = read_settings(base_settings_file)
        raw = default_settings.get("video_conversion_range")
        if raw:
            import json
            try:
                video_conversion_range = json.loads(raw)
            except (ValueError, TypeError):
                pass

    fps_hint = 30  # used only for the duration hint in the log; actual fps read from video later
    if video_conversion_range is not None:
        start_fr, end_fr = video_conversion_range
        n_frames = end_fr - start_fr
        logger.info(
            "Frame range: %s → %d frames (%d–%d) (~%.1f s at %d fps)",
            video_conversion_range, n_frames,
            start_fr, end_fr, n_frames / fps_hint, fps_hint,
        )
    else:
        logger.warning(
            "video_conversion_range not set in project config or default.settings "
            "— converting and rendering the full video."
        )
    meta_real_width        = effective_config["meta_real_width"]
    track_max_individuals  = effective_config["track_max_individuals"]
    individual_prefix      = effective_config["individual_prefix"]
    video_extension        = effective_config.get("video_extension", "MP4")

    video_file = videos_dir / f"{video_name}.{video_extension}"
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    pv_dir = tuning_dir / "2_pv"
    pv_dir.mkdir(parents=True, exist_ok=True)

    from ._background import copy_background_to_sweep
    copy_background_to_sweep(
        video_name=video_name,
        project_pv_dir=project_dir / "2_tracking" / "2_pv",
        sweep_pv_dir=pv_dir,
    )

    # --------------------------------------------------------------------------
    clip_files = []
    was_cancelled = False

    for threshold in thresholds:
        if _stop_event.is_set():
            logger.info("Sweep stopped by user request — skipping remaining thresholds.")
            was_cancelled = True
            break

        logger.info("=" * 50)
        logger.info("Threshold %d", threshold)
        logger.info("=" * 50)

        settings_file = tuning_dir / f"{video_name}_thresh{threshold:03d}.settings"
        csv_dir       = tuning_dir / f"csv_{video_name}_t{threshold:03d}"
        csv_dir.mkdir(parents=True, exist_ok=True)

        overrides = {
            "detect_threshold":      threshold,
            "meta_real_width":       meta_real_width,
            "track_max_individuals": track_max_individuals,
            "individual_prefix":     individual_prefix,
        }

        # Parameters exposed in the Tune / Parameters tabs. Only those actually
        # set in the resolved config are written; the rest fall through to
        # default.settings. track_max_individuals is included in the spec and
        # overwrites the value set above when configured there.
        overrides.update(collect_overrides(effective_config))

        if video_conversion_range is not None:
            overrides["video_conversion_range"] = video_conversion_range

        if animal_size_min is not None or animal_size_max is not None:
            resolved_min, resolved_max = animal_size_min, animal_size_max
            if resolved_min is None or resolved_max is None:
                # Only one side was given — read the missing side from
                # default.settings' existing detect_size_filter so we don't
                # silently drop the override entirely.
                from ._settings import read_settings
                import json
                default_settings = read_settings(base_settings_file)
                raw = default_settings.get("detect_size_filter")
                default_min, default_max = None, None
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if parsed and isinstance(parsed[0], list) and len(parsed[0]) == 2:
                            default_min, default_max = parsed[0]
                    except (ValueError, TypeError, IndexError):
                        pass
                if resolved_min is None:
                    resolved_min = default_min
                if resolved_max is None:
                    resolved_max = default_max
                logger.info(
                    "Only one animal size bound given (min=%s, max=%s) — "
                    "filling the other from default.settings: min=%s, max=%s",
                    animal_size_min, animal_size_max, resolved_min, resolved_max,
                )

            if resolved_min is not None and resolved_max is not None:
                size_filter_value = [[resolved_min, resolved_max]]
                overrides["detect_size_filter"] = size_filter_value
                overrides["track_size_filter"]  = size_filter_value
            else:
                logger.warning(
                    "Could not resolve a complete animal size range (min=%s, "
                    "max=%s) — detect_size_filter/track_size_filter left at "
                    "default.settings value.",
                    resolved_min, resolved_max,
                )

        # Apply masks if present for this video (video → batch → default)
        from ._masks import load_resolved_masks, masks_to_trex_string
        masks       = load_resolved_masks(video_name, project_config, masks_dir)
        has_include = bool(masks["include"])
        has_ignore  = bool(masks["ignore"])
        if has_include and has_ignore:
            logger.warning(
                "Both include and ignore masks found for '%s' (source: %s). "
                "TRex will ignore the ignore masks when include masks are set.",
                video_name, masks["source"],
            )
        if has_include:
            overrides["track_include"] = masks_to_trex_string(masks["include"])
        elif has_ignore:
            overrides["track_ignore"] = masks_to_trex_string(masks["ignore"])

        patch_settings(
            base_settings_file=base_settings_file,
            output_file=settings_file,
            overrides=overrides,
        )

        try:
            _run_tgrabs(
                video_file=video_file,
                settings_file=settings_file,
                video_name=video_name,
                pv_dir=pv_dir,
                pipeline_config=pipeline_config,
                video_conversion_range=video_conversion_range,
                detect_threshold=threshold,
            )

            pv_file = pv_dir / f"{video_name}.pv"
            if not pv_file.exists():
                logger.warning(
                    "TGrabs did not produce a .pv file for threshold=%d — skipping.", threshold
                )
                continue

            _run_trex(
                pv_file=pv_file,
                settings_file=settings_file,
                video_name=video_name,
                csv_dir=csv_dir,
                pipeline_config=pipeline_config,
            )
        except SweepCancelled:
            logger.info("Sweep stopped by user request mid-threshold (threshold=%d).", threshold)
            was_cancelled = True
            break

        _promote_csvs(csv_dir=csv_dir, video_name=video_name)
        _cleanup_trex_byproducts(pv_dir=pv_dir, csv_dir=csv_dir, video_name=video_name)

        clip_file = tuning_dir / f"{video_name}_t{threshold:03d}.mp4"
        _render_annotated_clip(
            video_name=video_name,
            csv_dir=csv_dir,
            video_file=video_file,
            video_conversion_range=video_conversion_range,
            meta_real_width=meta_real_width,
            trail_seconds=effective_config.get("trail_seconds", 1.0),
            threshold=threshold,
            output_file=clip_file,
        )
        clip_files.append(clip_file)

    if was_cancelled:
        logger.info(
            "Sweep cancelled. %d clip(s) completed before stopping.", len(clip_files)
        )
    _clear_sweep_stop()
    return clip_files


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


class SweepCancelled(Exception):
    """Raised when a sweep is stopped via request_sweep_stop()."""


def _run_subprocess_cancellable(command: str) -> None:
    """
    Run a shell command, registering it as the current cancellable process
    so request_sweep_stop() can terminate it immediately. Raises
    SweepCancelled if the process was terminated due to a stop request
    (rather than failing on its own), or subprocess.CalledProcessError for
    a genuine failure.
    """
    global _current_process

    process = subprocess.Popen(["bash", "-c", command])
    with _current_process_lock:
        _current_process = process

    return_code = process.wait()

    with _current_process_lock:
        if _current_process is process:
            _current_process = None

    if return_code != 0:
        if _stop_event.is_set():
            raise SweepCancelled("Sweep stopped by user request.")
        raise subprocess.CalledProcessError(return_code, command)


def _run_tgrabs(
    video_file: Path,
    settings_file: Path,
    video_name: str,
    pv_dir: Path,
    pipeline_config: dict,
    video_conversion_range: list | None,
    detect_threshold: int,
) -> None:
    range_flag = (
        f'-video_conversion_range "[{video_conversion_range[0]},{video_conversion_range[1]}]" '
        if video_conversion_range is not None
        else ""
    )
    command = (
        f"{_conda_prefix(pipeline_config)} && "
        f"trex -task convert "
        f'-i "{video_file}" '
        f'-s "{settings_file}" '
        f"{range_flag}"
        f'-detect_threshold "{detect_threshold}" '
        f'-o "{video_name}" '
        f'-d "{pv_dir}"'
    )
    logger.info("TGrabs: %s", command)
    _run_subprocess_cancellable(command)


def _run_trex(
    pv_file: Path,
    settings_file: Path,
    video_name: str,
    csv_dir: Path,
    pipeline_config: dict,
) -> None:
    command = (
        f"{_conda_prefix(pipeline_config)} && "
        f"trex -task track "
        f'-i "{pv_file}" '
        f'-s "{settings_file}" '
        f'-data_prefix "{video_name}" '
        f'-output_dir "{csv_dir}"'
    )
    logger.info("TRex: %s", command)
    _run_subprocess_cancellable(command)


def _promote_csvs(csv_dir: Path, video_name: str) -> None:
    """
    TRex writes CSVs into {csv_dir}/{video_name}/*.csv.
    Move them up into csv_dir directly and remove the now-empty subfolder.
    """
    trex_subdir = csv_dir / video_name
    if not trex_subdir.exists():
        logger.warning("Expected TRex output subfolder not found: %s", trex_subdir)
        return
    for csv_file in trex_subdir.iterdir():
        csv_file.rename(csv_dir / csv_file.name)
    trex_subdir.rmdir()
    logger.info("Promoted CSVs to %s", csv_dir)


def _cleanup_trex_byproducts(pv_dir: Path, csv_dir: Path, video_name: str) -> None:
    """
    Remove files TRex writes that are not needed downstream.
    Logs all deletions in a single line.
    """
    candidates = [
        pv_dir  / f"{video_name}.results",
        pv_dir  / f"{video_name}.settings",
        pv_dir  / f"{video_name}.settings.backup",
        csv_dir / f"{video_name}.results.meta",
    ]
    deleted = [f for f in candidates if f.exists()]
    for unwanted_file in deleted:
        unwanted_file.unlink()
    if deleted:
        logger.info(
            "Removed TRex byproducts: %s",
            ", ".join(f.name for f in deleted),
        )


# =============================================================================
# Frame annotation
# =============================================================================

def _render_annotated_clip(
    video_name: str,
    csv_dir: Path,
    video_file: Path,
    video_conversion_range: list | None,
    meta_real_width: float,
    trail_seconds: float,
    threshold: int,
    output_file: Path,
) -> None:
    """
    Render an annotated clip for one threshold.

    Each tracked blob is drawn with a per-individual coloured bounding box,
    a trail of the last trail_seconds of movement, and text labels:
      - individual ID (short suffix)
      - blob area in pixels and cm²
      - frame-to-frame centroid speed in cm/s
    """
    csv_files = sorted(csv_dir.glob(f"{video_name}_*.csv"))
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

    total_frames       = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame        = video_conversion_range[0] if video_conversion_range is not None else 0
    raw_end_frame      = video_conversion_range[1] if video_conversion_range is not None else -1
    resolved_end_frame = total_frames if raw_end_frame == -1 else raw_end_frame

    # OpenCV's avc1 (H.264) writer support is inconsistent across builds —
    # write with mp4v (broadly supported) and re-encode to H.264 via FFmpeg
    # afterward, since browsers require H.264 for native <video> playback.
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    colour_map      = _assign_individual_colours(trajectories_df)
    speed_lookup    = _compute_speeds(trajectories_df, fps, cm_per_pixel)
    trail_lookup    = _build_trail_lookup(trajectories_df)
    trail_window_us = int(trail_seconds * 1_000_000)
    n_frames        = resolved_end_frame - start_frame

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    from tqdm import tqdm
    for frame_index in tqdm(
        range(start_frame, resolved_end_frame),
        total=n_frames,
        desc=f"  threshold={threshold}",
        unit="fr",
        dynamic_ncols=True,
    ):
        success, frame = capture.read()
        if not success:
            break

        # TRex numbers frames relative to video_conversion_range start (0-indexed),
        # so subtract start_frame to look up the correct row in the CSV.
        csv_frame_index = frame_index - start_frame

        frame = _annotate_frame(
            frame=frame,
            frame_index=csv_frame_index,
            trajectories_df=trajectories_df,
            speed_lookup=speed_lookup,
            trail_lookup=trail_lookup,
            trail_window_us=trail_window_us,
            colour_map=colour_map,
            cm_per_pixel=cm_per_pixel,
        )

        cv2.rectangle(frame, (0, 0), (frame_width, 26), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"threshold={threshold}  frame={frame_index} (csv={csv_frame_index})",
            (6, 18), FONT, 0.5, COLOUR_TEXT, 1, cv2.LINE_AA,
        )

        writer.write(frame)

    capture.release()
    writer.release()
    logger.info("Annotated clip (mp4v): %s", output_file)

    _reencode_for_browser_playback(output_file)


# Module-level flag so the "FFmpeg not found" warning is logged once per
# process, not once per threshold during a sweep.
_FFMPEG_MISSING_WARNED = False


def _reencode_for_browser_playback(video_file: Path) -> None:
    """
    Re-encode an OpenCV-written (mp4v) clip to H.264 in place via FFmpeg,
    so it plays natively in browser <video> elements (Tune tab grid,
    Background tab viewer). If FFmpeg is not installed, the clip is left
    as mp4v — it still opens fine in VLC/desktop players, just not in-browser.
    """
    global _FFMPEG_MISSING_WARNED

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        if not _FFMPEG_MISSING_WARNED:
            logger.warning(
                "ffmpeg not found on PATH — clips will remain mp4v and will "
                "NOT play in the browser (Tune tab grid, Background tab "
                "viewer), though they still open fine in VLC/desktop players. "
                "Install ffmpeg (e.g. 'conda install ffmpeg' or "
                "'sudo apt install ffmpeg') to enable in-browser playback."
            )
            _FFMPEG_MISSING_WARNED = True
        return

    temp_file = video_file.with_suffix(".h264.mp4")
    command = [
        ffmpeg_path, "-y",
        "-i", str(video_file),
        "-vcodec", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",   # ensures broad browser compatibility
        "-loglevel", "error",
        str(temp_file),
    ]
    try:
        subprocess.run(command, check=True)
        temp_file.replace(video_file)
        logger.info("Re-encoded to H.264 for browser playback: %s", video_file.name)
    except subprocess.CalledProcessError as error:
        logger.warning(
            "FFmpeg re-encode failed for %s (%s). Clip remains mp4v — "
            "playable in VLC/desktop players but not in-browser.",
            video_file.name, error,
        )
        temp_file.unlink(missing_ok=True)


def _load_trajectories(csv_files: list[Path]) -> pd.DataFrame:
    """
    Load and concatenate per-individual CSVs; normalise and rename columns.

    TRex column name quirks handled here:
      - Leading '#' stripped, whitespace stripped, lowercased
      - 'x#wcentroid (cm)' → 'x'  (and any other x#... variant)
      - 'y#wcentroid (cm)' → 'y'  (and any other y#... variant)
      - 'timestamp'        → 'time'
    """
    import re

    individual_dfs = []
    for csv_file in csv_files:
        individual_df = pd.read_csv(csv_file)

        # Step 1: strip leading #, whitespace, lowercase
        individual_df.columns = [
            col.strip().lstrip("#").strip().lower()
            for col in individual_df.columns
        ]

        # Step 2: rename x/y variants and timestamp
        rename_map = {}
        for col in individual_df.columns:
            if re.match(r"^x[#\s]", col):
                rename_map[col] = "x"
            elif re.match(r"^y[#\s]", col):
                rename_map[col] = "y"
            elif col == "timestamp":
                rename_map[col] = "time"
        individual_df = individual_df.rename(columns=rename_map)

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

    Non-finite positions (TRex outputs inf when individual is not found)
    break the speed chain: previous_row is reset to None so no spurious
    speed is computed across a gap.

    Returns: {individual_id: {frame_index: speed_cm_per_s}}
    """
    speed_lookup = {}
    for individual_id, individual_df in trajectories_df.groupby("individual_id"):
        individual_df = individual_df.sort_values("frame")
        speeds        = {}
        previous_row  = None
        for _, row in individual_df.iterrows():
            frame_index = int(row["frame"])
            if not (np.isfinite(row.get("x", np.nan)) and np.isfinite(row.get("y", np.nan))):
                previous_row = None
                continue
            if previous_row is not None:
                dx_pixels      = row["x"] - previous_row["x"]
                dy_pixels      = row["y"] - previous_row["y"]
                distance_cm    = np.hypot(dx_pixels, dy_pixels) * cm_per_pixel
                speed_cm_per_s = distance_cm * fps
                speeds[frame_index] = speed_cm_per_s
            previous_row = row
        speed_lookup[individual_id] = speeds
    return speed_lookup


def _assign_individual_colours(trajectories_df: pd.DataFrame) -> dict:
    """
    Assign a BGR colour to each individual ID, cycling over INDIVIDUAL_COLOURS.
    Assignment is sorted by individual_id for deterministic results.

    Returns: {individual_id: (B, G, R)}
    """
    individual_ids = sorted(trajectories_df["individual_id"].unique())
    return {
        individual_id: INDIVIDUAL_COLOURS[index % len(INDIVIDUAL_COLOURS)]
        for index, individual_id in enumerate(individual_ids)
    }


def _build_trail_lookup(trajectories_df: pd.DataFrame) -> dict:
    """
    Precompute per-individual ordered list of valid (time_us, x, y) tuples,
    sorted ascending by time. Only finite positions are included.

    Returns: {individual_id: [(time_us, x, y), ...]}
    """
    trail_lookup = {}
    for individual_id, individual_df in trajectories_df.groupby("individual_id"):
        individual_df = individual_df.sort_values("time")
        valid_positions = [
            (int(row["time"]), float(row["x"]), float(row["y"]))
            for _, row in individual_df.iterrows()
            if np.isfinite(row.get("x", np.nan)) and np.isfinite(row.get("y", np.nan))
            and np.isfinite(row.get("time", np.nan))
        ]
        trail_lookup[individual_id] = valid_positions
    return trail_lookup


def _annotate_frame(
    frame: np.ndarray,
    frame_index: int,
    trajectories_df: pd.DataFrame,
    speed_lookup: dict,
    trail_lookup: dict,
    trail_window_us: int,
    colour_map: dict,
    cm_per_pixel: float,
) -> np.ndarray:
    """
    Draw trails, bounding boxes, and metric labels for all individuals in one frame.

    Trails are drawn first so bounding boxes and labels render on top.
    Each individual gets a distinct colour from colour_map.
    Trail segments are drawn only between consecutive valid positions;
    gaps (inf frames) cause the trail to break naturally.
    """
    frame_df = trajectories_df[trajectories_df["frame"] == frame_index]

    # ---- trails ----
    for _, row in frame_df.iterrows():
        if not (np.isfinite(row.get("time", np.nan))):
            continue

        individual_id   = row["individual_id"]
        current_time_us = int(row["time"])
        colour          = colour_map.get(individual_id, INDIVIDUAL_COLOURS[0])
        trail_positions = trail_lookup.get(individual_id, [])

        # Collect positions within the trail window up to (and including) current time
        window_positions = [
            (time_us, x, y)
            for time_us, x, y in trail_positions
            if current_time_us - trail_window_us <= time_us <= current_time_us
        ]

        # Draw line segments between consecutive points in the window.
        # A segment is only drawn if both endpoints are present in trail_positions
        # consecutively — gaps (missing frames with inf) are already excluded
        # because _build_trail_lookup only keeps finite positions.
        for segment_index in range(1, len(window_positions)):
            _, x_prev, y_prev = window_positions[segment_index - 1]
            _, x_curr, y_curr = window_positions[segment_index]
            cv2.line(
                frame,
                (int(x_prev), int(y_prev)),
                (int(x_curr), int(y_curr)),
                colour,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    # ---- bounding boxes and labels ----
    for _, row in frame_df.iterrows():
        if pd.isna(row.get("x")) or pd.isna(row.get("y")):
            continue
        if not (np.isfinite(row["x"]) and np.isfinite(row["y"])):
            continue

        cx = int(row["x"])
        cy = int(row["y"])
        blob_width_px  = int(row.get("blob_width",  20))
        blob_height_px = int(row.get("blob_height", 20))

        x1 = cx - blob_width_px  // 2
        y1 = cy - blob_height_px // 2
        x2 = cx + blob_width_px  // 2
        y2 = cy + blob_height_px // 2

        num_pixels    = row.get("num_pixels", blob_width_px * blob_height_px)
        area_cm2      = num_pixels * (cm_per_pixel ** 2)
        individual_id = row["individual_id"]
        colour        = colour_map.get(individual_id, INDIVIDUAL_COLOURS[0])

        speed_cm_per_s = speed_lookup.get(individual_id, {}).get(frame_index)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)

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
