"""
Background image computation for TRex.

TGrabs automatically uses a background image named average_{video_name}.png
if it exists in the same directory as the .pv output files (2_pv/ for full
runs, tuning/sweep_*/2_pv/ for sweep runs).

Unified model: a background image is computed by sampling n_images frames,
evenly spread between start_time and end_time (seconds), and combining them
pixel-wise using one of: mean, median, max, min.

A dedicated background video, if present ({video_name}_average.{ext} in
1_videos/), is treated identically to the main video — just pointed at a
different source file. Calling with n_images=1 degenerates to "the single
sampled frame", which covers the old quick-extraction use case.
"""

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VALID_METHODS = ("mean", "median", "max", "min")


# =============================================================================
# Core computation
# =============================================================================

def compute_background_image(
    video_file: Path,
    method: str,
    n_images: int = 10,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    """
    Compute a background image by sampling n_images frames evenly spread
    between start_time and end_time (seconds), combined pixel-wise.

    If start_time / end_time are None, the sampling range defaults to
    [1.0, duration - 1.0] seconds — trimming the first and last second to
    avoid camera shake right after pressing record / before stopping.
    If the user supplies explicit times, those exact bounds are used.

    method: one of 'mean', 'median', 'max', 'min'

    Returns the combined image as a BGR numpy array (uint8).
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    fps          = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    if total_frames == 0:
        capture.release()
        raise RuntimeError(f"Video has no frames: {video_file}")

    resolved_start = start_time if start_time is not None else min(1.0, duration_s / 2)
    resolved_end   = end_time   if end_time   is not None else max(duration_s - 1.0, resolved_start)

    if resolved_end <= resolved_start:
        capture.release()
        raise ValueError(
            f"end_time ({resolved_end:.2f}s) must be greater than "
            f"start_time ({resolved_start:.2f}s)."
        )

    sample_frame_indices = _evenly_spaced_frame_indices(
        start_frame=int(resolved_start * fps),
        end_frame=min(int(resolved_end * fps), total_frames - 1),
        n_images=n_images,
    )

    sampled_frames = []
    for frame_index in sample_frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            logger.warning("Could not read frame %d — skipping.", frame_index)
            continue
        sampled_frames.append(frame)
    capture.release()

    if not sampled_frames:
        raise RuntimeError(f"No frames could be sampled from {video_file}")

    stacked = np.stack(sampled_frames, axis=0).astype(np.float32)

    if method == "mean":
        combined = np.mean(stacked, axis=0)
    elif method == "median":
        combined = np.median(stacked, axis=0)
    elif method == "max":
        combined = np.max(stacked, axis=0)
    else:  # min
        combined = np.min(stacked, axis=0)

    return np.clip(combined, 0, 255).astype(np.uint8)


def save_background_image(image: np.ndarray, output_file: Path) -> None:
    """Save a computed background image as a PNG."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), image)
    logger.info("Saved background image: %s", output_file)


def background_image_path(video_name: str, pv_dir: Path) -> Path:
    """Return the expected path for a video's background image."""
    return pv_dir / f"average_{video_name}.png"


def resolve_background_source(
    video_name: str,
    video_extension: str,
    videos_dir: Path,
) -> tuple[Path, bool]:
    """
    Determine which video file to sample frames from for background
    computation: a dedicated {video_name}_average.{ext} if present,
    otherwise the main tracking video itself.

    Returns (source_video_file, is_dedicated_background_video).
    """
    dedicated_file = videos_dir / f"{video_name}_average.{video_extension}"
    if dedicated_file.exists():
        return dedicated_file, True
    main_file = videos_dir / f"{video_name}.{video_extension}"
    return main_file, False


def copy_background_to_sweep(
    video_name: str,
    project_pv_dir: Path,
    sweep_pv_dir: Path,
) -> None:
    """
    Copy average_{video_name}.png from the project's 2_pv/ into a sweep's
    2_pv/ subfolder so TGrabs picks it up there too.
    Does nothing if the image does not exist.
    """
    background_image_file = project_pv_dir / f"average_{video_name}.png"
    if background_image_file.exists():
        destination_file = sweep_pv_dir / background_image_file.name
        shutil.copy2(background_image_file, destination_file)
        logger.info("Copied background image to %s", destination_file)


# =============================================================================
# Helpers
# =============================================================================

def _evenly_spaced_frame_indices(start_frame: int, end_frame: int, n_images: int) -> list[int]:
    """
    Return n_images frame indices evenly spread between start_frame and
    end_frame (inclusive). Falls back gracefully if the range is too small
    to fit n_images distinct values.
    """
    if n_images <= 1:
        return [int((start_frame + end_frame) / 2)]
    if end_frame <= start_frame:
        return [start_frame]
    step = (end_frame - start_frame) / (n_images - 1)
    return [int(round(start_frame + i * step)) for i in range(n_images)]
