"""
_masks.py — mask file I/O, TRex serialisation, and circle auto-detection.

Mask files live in projects/{name}/masks/ and follow the naming convention:
    {video_name}_include_{n}.csv    ← one include polygon per file
    {video_name}_ignore_{n}.csv     ← one ignore polygon per file
    default_include_{n}.csv         ← project-default include polygon
    default_ignore_{n}.csv          ← project-default ignore polygon

Each CSV has two columns (no header): x, y — one vertex per row, in pixels
relative to the full-resolution source frame.

The pipeline (_sweep.py) reads only {video_name}_*.csv files. Default files
are only read by the mask editor app. Saving a video's masks always produces
per-video files; defaults are only updated via an explicit UI action.
"""

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


def load_default_masks(masks_dir: Path) -> dict:
    """
    Load project-default mask polygons.
    Returns {"include": [...], "ignore": [...]}
    """
    return {
        "include": _load_polygon_files(masks_dir, "default_include_*.csv"),
        "ignore":  _load_polygon_files(masks_dir, "default_ignore_*.csv"),
    }


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
