"""
Background image preparation for TRex.

TGrabs automatically uses a background image named average_{video_name}.png
if it exists in the same directory as the .pv output files (2_pv/ for full
runs, tuning/sweep_*/pv/ for sweep runs).

Workflow:
  - If {video_name}_average.{ext} exists in 1_videos/, extract its middle
    frame and save it as average_{video_name}.png in the given pv_dir.
  - If no background video is found, do nothing: TGrabs computes the
    background from the tracking video automatically.
"""

import logging
import shutil
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def prepare_background_image(
    video_name: str,
    video_extension: str,
    videos_dir: Path,
    pv_dir: Path,
) -> Path | None:
    """
    Look for {video_name}_average.{video_extension} in videos_dir.
    If found, extract its middle frame and save as average_{video_name}.png
    in pv_dir. TGrabs picks this up automatically by name.

    Returns the path to the saved image, or None if no background video exists.
    """
    background_video_file = videos_dir / f"{video_name}_average.{video_extension}"
    background_image_file = pv_dir / f"average_{video_name}.png"

    if not background_video_file.exists():
        logger.info(
            "No background video found for '%s' (expected: %s). "
            "TGrabs will compute the background automatically.",
            video_name, background_video_file,
        )
        return None

    if background_image_file.exists():
        logger.info(
            "Background image already exists at %s — skipping extraction.",
            background_image_file,
        )
        return background_image_file

    logger.info("Extracting background frame from %s", background_video_file)
    frame = _extract_middle_frame(background_video_file)
    cv2.imwrite(str(background_image_file), frame)
    logger.info("Saved background image: %s", background_image_file)
    return background_image_file


def copy_background_to_sweep(
    video_name: str,
    project_pv_dir: Path,
    sweep_pv_dir: Path,
) -> None:
    """
    Copy average_{video_name}.png from the project's 2_pv/ into a sweep's
    pv/ subfolder so TGrabs picks it up there too.
    Does nothing if the image does not exist.
    """
    background_image_file = project_pv_dir / f"average_{video_name}.png"
    if background_image_file.exists():
        destination_file = sweep_pv_dir / background_image_file.name
        shutil.copy2(background_image_file, destination_file)
        logger.info("Copied background image to %s", destination_file)


# ------------------------------------------------------------------------------

def _extract_middle_frame(video_file: Path):
    """Return the middle frame of a video as a numpy array."""
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise RuntimeError(f"Video has no frames: {video_file}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {total_frames // 2} from {video_file}"
        )
    return frame
