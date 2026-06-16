"""
new_project.py — scaffold a new tracking project from the template.

Usage:
    uv run python pipeline/new_project.py --name pain_killers
    uv run python pipeline/new_project.py --name pain_killers --path /data/projects

Creates:
    <path>/<name>/
        project.yaml          (filled with project name)
        1_videos/
        2_pv/
        3_csv_individual/
        4_csv_trial/
        tuning/

The project lives outside the repo by default (at --path), so it is never
accidentally committed. Pass --path inside the repo only if you know what
you are doing.
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------

PIPELINE_DIR  = Path(__file__).resolve().parent
REPO_ROOT     = PIPELINE_DIR.parent
TEMPLATE_FILE = REPO_ROOT / "project_template" / "project.yaml"

PROJECT_SUBDIRS = [
    "1_videos",
    "2_pv",
    "3_csv_individual",
    "4_csv_trial",
    "tuning",
]

# ---------------------------------------------------------------------------

def scaffold_project(project_name: str, projects_dir: Path) -> Path:
    project_dir = projects_dir / project_name

    if project_dir.exists():
        print(f"Project folder already exists: {project_dir}")
        print("Delete it first or choose a different name.")
        sys.exit(1)

    project_dir.mkdir(parents=True)

    # Copy template and substitute project name
    project_yaml_file = project_dir / "project.yaml"
    template_text = TEMPLATE_FILE.read_text()
    filled_text   = template_text.replace("__PROJECT_NAME__", project_name)
    project_yaml_file.write_text(filled_text)

    # Create subdirectories
    for subdir in PROJECT_SUBDIRS:
        (project_dir / subdir).mkdir()

    return project_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold a new insect tracking project."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Project name, e.g. pain_killers",
    )
    parser.add_argument(
        "--path",
        default=str(REPO_ROOT / "projects"),
        help=(
            "Parent directory in which to create the project folder. "
            "Default: <repo>/projects (gitignored). "
            "Override to place the project elsewhere, e.g. /data/projects"
        ),
    )
    args = parser.parse_args()

    projects_dir = Path(args.path).resolve()
    projects_dir.mkdir(parents=True, exist_ok=True)

    project_dir = scaffold_project(
        project_name=args.name,
        projects_dir=projects_dir,
    )

    print(f"Project created: {project_dir}")
    print("")
    print("Next steps:")
    print(f"  1. Edit {project_dir / 'project.yaml'}")
    print(f"  2. Place your video at {project_dir / '1_videos' / args.name}.MP4")
    print(f"  3. Run the sweep:")
    print(f"       uv run python pipeline/tune.py --project {project_dir} --trial <trial_name>")


if __name__ == "__main__":
    main()
