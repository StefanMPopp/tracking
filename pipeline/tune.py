"""
tune.py — Threshold sweep CLI for TRex tracking parameter selection.

Usage:
    uv run python pipeline/tune.py --project /path/to/project --trial pain_test
    uv run python pipeline/tune.py --project /path/to/project --trial pain_test --thresholds 20,28,35,50

For each candidate threshold the script:
  1. Prepares the background image (if a *_average.MP4 is present in 1_videos/)
  2. Runs TGrabs and TRex on the excerpt defined by video_conversion_range
  3. Renders an annotated clip with per-blob area (px, cm²) and speed (cm/s)
  4. Composes a side-by-side grid video of all threshold variants
  5. Opens the grid video and prompts you to accept a value or iterate

Confirmed thresholds are written back to project.yaml.
All sweep outputs are saved under tuning/sweep_{timestamp}/.
A cumulative tuning/tuning_log.json is maintained across all sweeps.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from _background import prepare_background_image
from _sweep import run_sweep

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
            f"  uv run python pipeline/new_project.py --name <name> --path <parent_dir>"
        )
    return yaml.safe_load(project_yaml_file.read_text())


def save_project_config(project_dir: Path, project_config: dict) -> None:
    project_yaml_file = project_dir / "project.yaml"
    project_yaml_file.write_text(
        yaml.dump(project_config, default_flow_style=False, sort_keys=False)
    )


def resolve_trial_config(project_config: dict, trial: str) -> dict:
    """
    Return effective config for a trial: project-level values with any
    trial-specific overrides applied on top.
    """
    effective_config = dict(project_config)
    trial_overrides  = (project_config.get("trial_overrides") or {}).get(trial, {})
    if trial_overrides:
        logger.info("Applying per-trial overrides for '%s': %s", trial, trial_overrides)
        effective_config.update(trial_overrides)
    return effective_config


# =============================================================================
# Tuning log
# =============================================================================

def append_tuning_log(
    tuning_dir: Path,
    trial: str,
    sweep_dir: Path,
    thresholds_tested: list,
    confirmed_threshold: int | None,
    scope: str | None,
) -> None:
    log_file = tuning_dir / "tuning_log.json"
    existing_entries = json.loads(log_file.read_text()) if log_file.exists() else []
    existing_entries.append({
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
        "trial":               trial,
        "sweep_dir":           str(sweep_dir),
        "thresholds_tested":   thresholds_tested,
        "confirmed_threshold": confirmed_threshold,
        "scope":               scope,
    })
    log_file.write_text(json.dumps(existing_entries, indent=2))
    logger.info("Tuning log updated: %s", log_file)


# =============================================================================
# User interaction
# =============================================================================

def open_video(video_file: Path) -> None:
    """Open a video with the system default player (non-blocking)."""
    if not video_file.exists():
        logger.warning("Video not found: %s", video_file)
        return
    try:
        subprocess.Popen(["xdg-open", str(video_file)])
    except FileNotFoundError:
        logger.info("Could not auto-open video. Open manually:\n  %s", video_file)


def prompt_decision(current_thresholds: list, trial: str) -> tuple:
    """
    Prompt the user after reviewing the grid video.

    Returns one of:
        ("accept",  (confirmed_value: int, scope: str))   scope = "project" | "trial"
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
                f"Save as [p]roject-wide default or [t]rial override for '{trial}'? "
                f"[p/t, default=p]: "
            ).strip().lower() or "p"
            scope = "trial" if scope_raw == "t" else "project"
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
        help="Path to the project folder, e.g. /home/stefan/tracking/projects/pain_killers",
    )
    parser.add_argument(
        "--trial",
        required=True,
        help="Trial name without extension, e.g. pain_test",
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
    effective_config = resolve_trial_config(project_config, args.trial)

    if args.thresholds:
        try:
            effective_config["sweep_thresholds"] = [
                int(v.strip()) for v in args.thresholds.split(",")
            ]
        except ValueError:
            logger.error("--thresholds must be comma-separated integers.")
            sys.exit(1)

    tuning_dir = project_dir / "tuning"
    tuning_dir.mkdir(exist_ok=True)

    # Prepare background image once before any sweep iteration
    prepare_background_image(
        trial=args.trial,
        videos_dir=project_dir / "1_videos",
        pv_dir=project_dir / "2_pv",
    )

    # ------------------------------------------------------------------
    # Sweep loop
    # ------------------------------------------------------------------
    current_thresholds    = list(effective_config["sweep_thresholds"])
    all_thresholds_tested = []

    while True:
        effective_config["sweep_thresholds"] = current_thresholds
        all_thresholds_tested.extend(current_thresholds)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = tuning_dir / f"sweep_{timestamp}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        grid_video_file = run_sweep(
            trial=args.trial,
            project_dir=project_dir,
            sweep_dir=sweep_dir,
            base_settings_file=BASE_SETTINGS_FILE,
            pipeline_config=pipeline_config,
            effective_config=effective_config,
        )

        open_video(grid_video_file)

        decision, payload = prompt_decision(
            current_thresholds=current_thresholds,
            trial=args.trial,
        )

        # - - - quit - - -
        if decision == "quit":
            append_tuning_log(
                tuning_dir=tuning_dir,
                trial=args.trial,
                sweep_dir=sweep_dir,
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

            if scope == "trial":
                if "trial_overrides" not in project_config:
                    project_config["trial_overrides"] = {}
                if args.trial not in project_config["trial_overrides"]:
                    project_config["trial_overrides"][args.trial] = {}
                project_config["trial_overrides"][args.trial][
                    "confirmed_detect_threshold"
                ] = confirmed_threshold
                logger.info(
                    "Saved detect_threshold=%d as override for trial '%s'.",
                    confirmed_threshold, args.trial,
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
                trial=args.trial,
                sweep_dir=sweep_dir,
                thresholds_tested=all_thresholds_tested,
                confirmed_threshold=confirmed_threshold,
                scope=scope,
            )
            break


if __name__ == "__main__":
    main()
