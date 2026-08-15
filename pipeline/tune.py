"""
tune.py — Threshold sweep CLI for TRex tracking parameter selection.

Usage:
    uv run python pipeline/tune.py --project /path/to/project --video pain_test
    uv run python pipeline/tune.py --project /path/to/project --video pain_test --thresholds 20,28,35,50

For each candidate threshold the script:
  1. Prepares the background image (if a {video}_average.{ext} is present in 1_videos/)
  2. Runs TGrabs and TRex on the excerpt defined by video_conversion_range
  3. Renders an annotated clip with per-blob area (px, cm²) and speed (cm/s)
  4. Opens each produced clip and prompts you to accept a value or iterate

All outputs go directly into tuning/ (flat structure, no timestamped subfolders).
Confirmed thresholds are written back to project.yaml.
A cumulative tuning/tuning_log.json is maintained across all sweeps.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


if __package__ in (None, ""):
    raise SystemExit(
        "This module is part of the 'pipeline' package and cannot be run as a\n"
        "standalone script (its imports are relative).\n\n"
        "Use instead:\n"
        "    uv run tracker --project <project_path>\n"
        "or, to run this module directly:\n"
        "    uv run python -m pipeline.tune --project <project_path>\n"
    )

from ._background import background_image_path
from ._resolve import resolve_effective_config
from ._sweep import run_sweep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PIPELINE_DIR         = Path(__file__).resolve().parent
BASE_SETTINGS_FILE   = PIPELINE_DIR / "default.settings"
PIPELINE_CONFIG_FILE = PIPELINE_DIR / "pipeline.yaml"


# =============================================================================
# Config helpers
# =============================================================================

def load_pipeline_config() -> dict:
    if not PIPELINE_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"pipeline.yaml not found at {PIPELINE_CONFIG_FILE}.\n"
            f"Copy {PIPELINE_DIR / 'pipeline.yaml.example'} to pipeline.yaml "
            f"and edit it for this machine."
        )
    return yaml.safe_load(PIPELINE_CONFIG_FILE.read_text())


def load_project_config(project_dir: Path) -> dict:
    project_yaml_file = project_dir / "project.yaml"
    if not project_yaml_file.exists():
        raise FileNotFoundError(
            f"project.yaml not found at {project_yaml_file}.\n"
            f"Create a project first:\n"
            f"  uv run tracker new-project --name <name> --path <parent_dir>"
        )
    return yaml.safe_load(project_yaml_file.read_text())


def save_project_config(project_dir: Path, project_config: dict) -> None:
    project_yaml_file = project_dir / "project.yaml"
    project_yaml_file.write_text(
        yaml.dump(project_config, default_flow_style=False, sort_keys=False)
    )


# resolve_effective_config is imported from _resolve (shared with app.py tabs)
# and layers: project-level → batch → video_overrides.
# video_conversion_range is intentionally NOT defaulted at any layer. If absent
# everywhere, it is left unset and _sweep.py will not pass -video_conversion_range
# to TGrabs, letting default.settings govern the range.


# =============================================================================
# Tuning log
# =============================================================================

def append_tuning_log(
    tuning_dir: Path,
    video_name: str,
    thresholds_tested: list,
    confirmed_threshold: int | None,
    scope: str | None,
) -> None:
    log_file         = tuning_dir / "tuning_log.json"
    existing_entries = json.loads(log_file.read_text()) if log_file.exists() else []
    existing_entries.append({
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
        "video":               video_name,
        "thresholds_tested":   thresholds_tested,
        "confirmed_threshold": confirmed_threshold,
        "scope":               scope,
    })
    log_file.write_text(json.dumps(existing_entries, indent=2))
    logger.info("Tuning log updated: %s", log_file)


# =============================================================================
# Video playback
# =============================================================================

def open_clips(clip_files: list[Path]) -> None:
    """
    Open each annotated clip with the system video player, suppressing
    player stderr/stdout to avoid VLC noise in the terminal.
    Tries 'vlc' first (direct, suppressible), falls back to 'xdg-open'.
    """
    player = shutil.which("vlc") or shutil.which("xdg-open")
    if player is None:
        logger.info("No video player found. Open clips manually:")
        for clip_file in clip_files:
            logger.info("  %s", clip_file)
        return

    for clip_file in clip_files:
        if not clip_file.exists():
            logger.warning("Clip not found: %s", clip_file)
            continue
        subprocess.Popen(
            [player, str(clip_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Opened: %s", clip_file)


# =============================================================================
# User interaction
# =============================================================================

def prompt_decision(current_thresholds: list, video_name: str) -> tuple:
    """
    Prompt the user after reviewing the annotated clips.

    Returns one of:
        ("accept",  (confirmed_value: int, scope: str))   scope = "project" | "video"
        ("resweep", new_thresholds: list)
        ("quit",    None)
    """
    print("\n" + "=" * 60)
    print(f"Sweep complete. Thresholds tested: {current_thresholds}")
    print("=" * 60)
    print("  [a]  Accept a threshold value")
    print("  [r]  Re-sweep with different threshold values")
    print("  [q]  Quit without saving")

    while True:
        choice = input("\nChoice [a/r/q]: ").strip().lower()

        if choice == "q":
            return "quit", None

        if choice == "a":
            raw = input("Threshold value to confirm: ").strip()
            try:
                confirmed_value = int(raw)
            except ValueError:
                print(f"  '{raw}' is not an integer. Try again.")
                continue
            scope_raw = input(
                f"Save as [p]roject-wide default or [v]ideo override for '{video_name}'? "
                f"[p/v, default=p]: "
            ).strip().lower() or "p"
            scope = "video" if scope_raw == "v" else "project"
            return "accept", (confirmed_value, scope)

        if choice == "r":
            raw = input(
                "New threshold values (comma-separated integers, e.g. 20,28,35): "
            ).strip()
            try:
                new_thresholds = [int(v.strip()) for v in raw.split(",")]
            except ValueError:
                print("  Could not parse values. Use integers separated by commas.")
                continue
            return "resweep", new_thresholds

        print("  Enter a, r, or q.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a detect_threshold sweep and select TRex tracking parameters."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Path to the project folder, e.g. /path/to/projects/pain_killers",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video name without extension, e.g. pain_test",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Comma-separated threshold values, overriding project.yaml. "
            "E.g. --thresholds 15,25,35,50"
        ),
    )
    args = parser.parse_args()

    project_dir = Path(args.project).resolve()
    if not project_dir.exists():
        logger.error("Project directory not found: %s", project_dir)
        sys.exit(1)

    pipeline_config  = load_pipeline_config()
    project_config   = load_project_config(project_dir)
    effective_config = resolve_effective_config(args.video, project_config)

    if args.thresholds:
        try:
            effective_config["sweep_thresholds"] = [
                int(v.strip()) for v in args.thresholds.split(",")
            ]
        except ValueError:
            logger.error("--thresholds must be comma-separated integers.")
            sys.exit(1)

    tuning_dir = project_dir / "2_tracking" / "tuning"
    tuning_dir.mkdir(exist_ok=True)

    video_extension = effective_config.get("video_extension", "MP4")
    background_image_file = background_image_path(args.video, project_dir / "2_tracking" / "2_pv")
    if not background_image_file.exists():
        logger.info(
            "No background image found for '%s'. TGrabs will compute one "
            "automatically, or run: uv run python pipeline/background.py "
            "--project %s --video %s",
            args.video, args.project, args.video,
        )

    # ==========================================================================
    # Sweep loop
    # ==========================================================================
    current_thresholds    = list(effective_config["sweep_thresholds"])
    all_thresholds_tested = []

    while True:
        effective_config["sweep_thresholds"] = current_thresholds
        all_thresholds_tested.extend(current_thresholds)

        clip_files = run_sweep(
            video_name=args.video,
            project_dir=project_dir,
            videos_dir=project_dir / "2_tracking" / "1_videos",
            masks_dir=project_dir / "2_tracking" / "masks",
            tuning_dir=tuning_dir,
            base_settings_file=BASE_SETTINGS_FILE,
            pipeline_config=pipeline_config,
            effective_config=effective_config,
            project_config=project_config,
        )

        open_clips(clip_files)

        decision, payload = prompt_decision(
            current_thresholds=current_thresholds,
            video_name=args.video,
        )

        # - - - quit - - -
        if decision == "quit":
            append_tuning_log(
                tuning_dir=tuning_dir,
                video_name=args.video,
                thresholds_tested=all_thresholds_tested,
                confirmed_threshold=None,
                scope=None,
            )
            logger.info("Exiting without saving a confirmed threshold.")
            break

        # - - - re-sweep - - -
        if decision == "resweep":
            current_thresholds = payload
            logger.info("Re-sweeping with thresholds: %s", current_thresholds)
            continue

        # - - - accept - - -
        if decision == "accept":
            confirmed_threshold, scope = payload

            if scope == "video":
                if "video_overrides" not in project_config:
                    project_config["video_overrides"] = {}
                if args.video not in project_config["video_overrides"]:
                    project_config["video_overrides"][args.video] = {}
                project_config["video_overrides"][args.video][
                    "confirmed_detect_threshold"
                ] = confirmed_threshold
                logger.info(
                    "Saved detect_threshold=%d as override for video '%s'.",
                    confirmed_threshold, args.video,
                )
            else:
                project_config["confirmed_detect_threshold"] = confirmed_threshold
                logger.info(
                    "Saved detect_threshold=%d as project-wide default.",
                    confirmed_threshold,
                )

            save_project_config(project_dir, project_config)
            append_tuning_log(
                tuning_dir=tuning_dir,
                video_name=args.video,
                thresholds_tested=all_thresholds_tested,
                confirmed_threshold=confirmed_threshold,
                scope=scope,
            )
            break


if __name__ == "__main__":
    main()
