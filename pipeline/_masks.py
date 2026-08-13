"""
_masks.py — mask file I/O, TRex serialisation, and circle auto-detection.

Mask files live in projects/{name}/masks/ and follow the naming convention:
    {video_name}_include_{n}.csv    ← one include polygon per file
    {video_name}_ignore_{n}.csv     ← one ignore polygon per file
    {batch_name}_include_{n}.csv    ← shared across all videos in that batch
    {batch_name}_ignore_{n}.csv
    default_include_{n}.csv         ← project-wide default include polygon
    default_ignore_{n}.csv          ← project-wide default ignore polygon

Each CSV has two columns (no header): x, y — one vertex per row, in pixels
relative to the full-resolution source frame.

Resolution for a given video (see load_resolved_masks): video-specific masks
win if present; otherwise the masks of the batch containing the video (see
_resolve.py); otherwise the project-wide defaults. Tiers are not merged —
the first tier with any shapes wins entirely.

Saving a video's masks always produces per-video files. Batch and default
files are only written via explicit UI actions ("save as batch defaults" /
"save as project defaults").
"""

import base64
import logging
import math
import re
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

POLYGON_VERTICES_PER_CIRCLE = 32   # vertices used to approximate a circle/oval


# =============================================================================
# File I/O
# =============================================================================

def load_masks_for_video(video_name: str, masks_dir: Path) -> dict:
    """
    Load all per-video mask polygons from masks_dir.
    Returns {"include": [polygon, ...], "ignore": [polygon, ...]}
    where each polygon is a list of [x, y] pairs (ints).
    Only reads {video_name}_include_*.csv and {video_name}_ignore_*.csv.
    """
    return {
        "include": _load_polygon_files(masks_dir, f"{video_name}_include_*.csv"),
        "ignore":  _load_polygon_files(masks_dir, f"{video_name}_ignore_*.csv"),
    }


def load_masks_for_batch(batch_name: str, masks_dir: Path) -> dict:
    """
    Load all batch-level mask polygons from masks_dir.
    Returns {"include": [...], "ignore": [...]}
    Only reads {batch_name}_include_*.csv and {batch_name}_ignore_*.csv.
    """
    return {
        "include": _load_polygon_files(masks_dir, f"{batch_name}_include_*.csv"),
        "ignore":  _load_polygon_files(masks_dir, f"{batch_name}_ignore_*.csv"),
    }


def load_default_masks(masks_dir: Path) -> dict:
    """
    Load project-default mask polygons.
    Returns {"include": [...], "ignore": [...]}
    """
    return {
        "include": _load_polygon_files(masks_dir, "default_include_*.csv"),
        "ignore":  _load_polygon_files(masks_dir, "default_ignore_*.csv"),
    }


def load_resolved_masks(video_name: str, project_config: dict, masks_dir: Path) -> dict:
    """
    Resolve which mask files apply to video_name, following the hierarchy:
        1. {video_name}_*.csv       (video-specific masks)
        2. {batch_name}_*.csv       (the batch containing this video, if any)
        3. default_*.csv            (project-wide default masks)

    The first tier with ANY shapes (include or ignore) wins entirely — tiers
    are not merged, since a partial mix of e.g. video-specific include shapes
    with batch-level ignore shapes would be ambiguous to reason about.

    Returns {"include": [...], "ignore": [...], "source": "video"|"batch"|"default"|"none"}
    """
    from ._resolve import resolve_batch_for_video

    video_masks = load_masks_for_video(video_name, masks_dir)
    if video_masks["include"] or video_masks["ignore"]:
        return {**video_masks, "source": "video"}

    batch_name = resolve_batch_for_video(video_name, project_config)
    if batch_name is not None:
        batch_masks = load_masks_for_batch(batch_name, masks_dir)
        if batch_masks["include"] or batch_masks["ignore"]:
            logger.info(
                "No video-specific masks for '%s' — using batch '%s' masks.",
                video_name, batch_name,
            )
            return {**batch_masks, "source": "batch"}

    default_masks = load_default_masks(masks_dir)
    if default_masks["include"] or default_masks["ignore"]:
        logger.info(
            "No video-specific or batch masks for '%s' — using project defaults.",
            video_name,
        )
        return {**default_masks, "source": "default"}

    return {"include": [], "ignore": [], "source": "none"}


def save_polygon(
    polygon: list,
    prefix: str,
    mask_type: str,
    masks_dir: Path,
) -> str:
    """
    Save one polygon to a new numbered CSV file.

    prefix:    video_name (e.g. 'pain_test') or 'default'
    mask_type: 'include' or 'ignore'

    Numbering starts at 1 and uses the next available integer
    (fills gaps to avoid ambiguity after deletions).

    Returns the filename (not full path).
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(masks_dir.glob(f"{prefix}_{mask_type}_*.csv"))
    existing_numbers = set()
    for existing_file in existing:
        match = re.search(r"_(\d+)\.csv$", existing_file.name)
        if match:
            existing_numbers.add(int(match.group(1)))

    number = 1
    while number in existing_numbers:
        number += 1

    output_file = masks_dir / f"{prefix}_{mask_type}_{number}.csv"
    lines = [f"{int(x)},{int(y)}" for x, y in polygon]
    output_file.write_text("\n".join(lines) + "\n")
    logger.info("Saved polygon: %s (%d vertices)", output_file.name, len(polygon))
    return output_file.name


def delete_polygon(filename: str, masks_dir: Path) -> None:
    """
    Delete a polygon CSV file. Logs a warning if not found.
    Remaining files are NOT renumbered — gaps in numbering are acceptable
    and avoid any risk of accidentally overwriting the wrong file.
    """
    target_file = masks_dir / filename
    if target_file.exists():
        target_file.unlink()
        logger.info("Deleted polygon file: %s", filename)
    else:
        logger.warning("Polygon file not found for deletion: %s", filename)


def save_default_masks(shapes: list, masks_dir: Path) -> None:
    """
    Overwrite all default_*.csv files with the given shapes.
    Existing default files are deleted first to avoid stale files.

    shapes: list of dicts with keys 'type' ('include'|'ignore') and
            'vertices' ([[x,y], ...])
    """
    for old_file in masks_dir.glob("default_*.csv"):
        old_file.unlink()

    for shape in shapes:
        save_polygon(
            polygon=shape["vertices"],
            prefix="default",
            mask_type=shape["type"],
            masks_dir=masks_dir,
        )
    logger.info("Saved %d shapes as project defaults.", len(shapes))


def save_batch_masks(shapes: list, batch_name: str, masks_dir: Path) -> None:
    """
    Overwrite all {batch_name}_*.csv files with the given shapes.
    Existing batch files are deleted first to avoid stale files.

    shapes: list of dicts with keys 'type' ('include'|'ignore') and
            'vertices' ([[x,y], ...])
    """
    for old_file in masks_dir.glob(f"{batch_name}_include_*.csv"):
        old_file.unlink()
    for old_file in masks_dir.glob(f"{batch_name}_ignore_*.csv"):
        old_file.unlink()

    for shape in shapes:
        save_polygon(
            polygon=shape["vertices"],
            prefix=batch_name,
            mask_type=shape["type"],
            masks_dir=masks_dir,
        )
    logger.info("Saved %d shapes as defaults for batch '%s'.", len(shapes), batch_name)


# =============================================================================
# TRex serialisation
# =============================================================================

def masks_to_trex_string(polygons: list) -> str:
    """
    Serialise a list of polygons to TRex settings format.

    TRex format: [[[x0,y0],[x1,y1],...],[[x0,y0],...]]
    Coordinates must be integers.
    """
    polygon_strings = []
    for polygon in polygons:
        vertex_strings = [f"[{int(x)},{int(y)}]" for x, y in polygon]
        polygon_strings.append("[" + ",".join(vertex_strings) + "]")
    return "[" + ",".join(polygon_strings) + "]"


# =============================================================================
# Circle auto-detection
# =============================================================================

def detect_circles(
    frame: np.ndarray,
    diameter_cm: float,
    thickness_cm: float,
    meta_real_width: float,
    expected_count: int,
    hue_center: int,
    hue_tolerance: int,
    saturation_min: int = 80,
    value_max: int = 120,
) -> list:
    """
    Detect hollow circles of known physical size and colour in a frame.

    Parameters
    ----------
    frame           : BGR image (full resolution)
    diameter_cm     : outer diameter of the circle in cm
    thickness_cm    : ring thickness in cm
    meta_real_width : physical width of the full frame in cm
    expected_count  : number of circles to find
    hue_center      : centre of the target hue in OpenCV HSV (0–179)
    hue_tolerance   : ± tolerance around hue_center
    saturation_min  : minimum saturation for colour mask
    value_max       : maximum value (brightness) — low = dark colours

    Returns a list of polygons (each a list of [x, y] pairs), up to
    expected_count, sorted by detection confidence (area proximity to ideal).
    """
    frame_width  = frame.shape[1]
    frame_height = frame.shape[0]
    px_per_cm    = frame_width / meta_real_width

    outer_radius_px = (diameter_cm / 2.0) * px_per_cm
    thickness_px    = thickness_cm * px_per_cm
    inner_radius_px = outer_radius_px - thickness_px

    # Expected contour area: annulus = π(R² - r²)
    ideal_area = math.pi * (outer_radius_px ** 2 - inner_radius_px ** 2)
    # Allow ±50% area tolerance
    area_min = ideal_area * 0.5
    area_max = ideal_area * 2.0

    # --- Colour mask (HSV) ---
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = (hue_center - hue_tolerance) % 180
    upper_hue = (hue_center + hue_tolerance) % 180

    if lower_hue <= upper_hue:
        colour_mask = cv2.inRange(
            hsv_frame,
            (lower_hue, saturation_min, 0),
            (upper_hue, 255, value_max),
        )
    else:
        # Hue range wraps around 0/180
        mask_low  = cv2.inRange(hsv_frame, (lower_hue, saturation_min, 0), (179, 255, value_max))
        mask_high = cv2.inRange(hsv_frame, (0, saturation_min, 0), (upper_hue, 255, value_max))
        colour_mask = cv2.bitwise_or(mask_low, mask_high)

    # --- Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, kernel)
    colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN,  kernel)

    # --- Find contours ---
    contours, _ = cv2.findContours(
        colour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Filter by area and circularity ---
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (area_min <= area <= area_max):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < 0.5:   # hollow circles are less circular than filled ones
            continue
        candidates.append((abs(area - ideal_area), contour))

    # Sort by proximity to ideal area, take top expected_count
    candidates.sort(key=lambda pair: pair[0])
    best_contours = [contour for _, contour in candidates[:expected_count]]

    if len(best_contours) < expected_count:
        logger.warning(
            "Auto-detect found %d of %d expected circles.",
            len(best_contours), expected_count,
        )

    # --- Convert to polygon approximations ---
    polygons = []
    for contour in best_contours:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        polygon = _circle_to_polygon(cx, cy, radius, POLYGON_VERTICES_PER_CIRCLE)
        polygons.append(polygon)

    return polygons


def detect_circles_debug(
    frame: np.ndarray,
    diameter_cm: float,
    thickness_cm: float,
    meta_real_width: float,
    expected_count: int,
    hue_center: int,
    hue_tolerance: int,
    saturation_min: int = 80,
    value_max: int = 120,
) -> dict:
    """
    Same detection pipeline as detect_circles, but returns every intermediate
    stage and a per-contour breakdown of why each candidate was accepted or
    rejected, for diagnosing "nothing detected" cases.

    Returns a dict:
        {
            "colour_mask_image":   base64 PNG (raw HSV threshold result),
            "cleaned_mask_image":  base64 PNG (after morphological close/open),
            "contours_image":      base64 PNG (frame annotated with every
                                    contour found, colour-coded green=accepted,
                                    red=rejected, with reason + measured
                                    area/circularity as text),
            "ideal_area":          float,
            "area_min":            float,
            "area_max":            float,
            "contour_details":     [
                {"area": float, "circularity": float, "accepted": bool,
                 "reason": str}, ...
            ],
        }
    """
    frame_width  = frame.shape[1]
    px_per_cm    = frame_width / meta_real_width

    outer_radius_px = (diameter_cm / 2.0) * px_per_cm
    thickness_px    = thickness_cm * px_per_cm
    inner_radius_px = outer_radius_px - thickness_px

    ideal_area = math.pi * (outer_radius_px ** 2 - inner_radius_px ** 2)
    area_min   = ideal_area * 0.5
    area_max   = ideal_area * 2.0

    # --- Colour mask (HSV) — identical logic to detect_circles ---
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = (hue_center - hue_tolerance) % 180
    upper_hue = (hue_center + hue_tolerance) % 180

    if lower_hue <= upper_hue:
        colour_mask = cv2.inRange(
            hsv_frame, (lower_hue, saturation_min, 0), (upper_hue, 255, value_max),
        )
    else:
        mask_low  = cv2.inRange(hsv_frame, (lower_hue, saturation_min, 0), (179, 255, value_max))
        mask_high = cv2.inRange(hsv_frame, (0, saturation_min, 0), (upper_hue, 255, value_max))
        colour_mask = cv2.bitwise_or(mask_low, mask_high)

    colour_mask_image = cv2.cvtColor(colour_mask, cv2.COLOR_GRAY2BGR)

    # --- Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN,  kernel)
    cleaned_mask_image = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

    # --- Find contours and classify each one ---
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours_image = frame.copy()
    contour_details = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        if not (area_min <= area <= area_max):
            accepted = False
            reason = f"area {area:.0f} outside [{area_min:.0f}, {area_max:.0f}]"
        elif circularity < 0.5:
            accepted = False
            reason = f"circularity {circularity:.2f} < 0.5"
        else:
            accepted = True
            reason = "accepted"

        contour_details.append({
            "area": float(area),
            "circularity": float(circularity),
            "accepted": accepted,
            "reason": reason,
        })

        colour = (50, 220, 50) if accepted else (50, 50, 220)
        cv2.drawContours(contours_image, [contour], -1, colour, 2)

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            label_x = int(moments["m10"] / moments["m00"])
            label_y = int(moments["m01"] / moments["m00"])
            label = f"a={area:.0f} c={circularity:.2f}"
            cv2.putText(
                contours_image, label, (label_x - 40, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return {
        "colour_mask_image":  _encode_image_b64(colour_mask_image),
        "cleaned_mask_image": _encode_image_b64(cleaned_mask_image),
        "contours_image":     _encode_image_b64(contours_image),
        "ideal_area":         float(ideal_area),
        "area_min":           float(area_min),
        "area_max":           float(area_max),
        "contour_details":    contour_details,
    }


# =============================================================================
# Helpers
# =============================================================================

def _load_polygon_files(masks_dir: Path, glob_pattern: str) -> list:
    """Read all CSV files matching glob_pattern, return list of polygons."""
    if not masks_dir.exists():
        return []
    polygon_files = sorted(masks_dir.glob(glob_pattern))
    polygons = []
    for polygon_file in polygon_files:
        polygon = _read_polygon_csv(polygon_file)
        if polygon:
            polygons.append(polygon)
    return polygons


def _read_polygon_csv(polygon_file: Path) -> list:
    """Read a polygon CSV (x,y per row, no header). Returns [[x,y], ...]."""
    vertices = []
    for line in polygon_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            logger.warning("Unexpected format in %s: '%s'", polygon_file.name, line)
            continue
        try:
            vertices.append([int(float(parts[0])), int(float(parts[1]))])
        except ValueError:
            logger.warning("Could not parse coordinates in %s: '%s'", polygon_file.name, line)
    return vertices


def _circle_to_polygon(cx: float, cy: float, radius: float, n_vertices: int) -> list:
    """Approximate a circle as an n-sided polygon."""
    return [
        [
            int(cx + radius * math.cos(2 * math.pi * i / n_vertices)),
            int(cy + radius * math.sin(2 * math.pi * i / n_vertices)),
        ]
        for i in range(n_vertices)
    ]


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode a BGR image as a base64 PNG string for embedding in JSON."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")
