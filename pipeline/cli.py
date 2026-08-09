"""
cli.py — console entry point for the `tracker` command.

    tracker --project <dir> [...]        launch the browser app
    tracker new-project --name X [...]   scaffold a new project repo

`new-project` is dispatched here; everything else falls through to app.py's
own argument parser, so app flags (--video, --mode, --tab, --port) work
unchanged.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from ._scaffold import (
    TRACKER_REPO_URL,
    print_next_steps,
    resolve_tracker_tag,
    scaffold_project,
)

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT   = PACKAGE_DIR.parent


def run_new_project(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="tracker new-project",
        description="Scaffold a new experiment project that uses this tracker.",
    )
    parser.add_argument("--name", required=True,
                        help="Project name, e.g. pain_killers")
    parser.add_argument("--path", default=str(Path.home() / "projects"),
                        help="Parent directory for the new project folder "
                             "(default: ~/projects)")
    args = parser.parse_args(argv)

    try:
        tracker_tag = resolve_tracker_tag(REPO_ROOT, __version__)
    except RuntimeError as error:
        print("")
        print("Cannot create a project:")
        print("")
        print(str(error))
        sys.exit(1)

    projects_dir = Path(args.path).expanduser().resolve()
    projects_dir.mkdir(parents=True, exist_ok=True)

    try:
        project_dir = scaffold_project(
            project_name=args.name,
            projects_dir=projects_dir,
            tracker_tag=tracker_tag,
            project_template_file=REPO_ROOT / "project_template" / "project.yaml",
        )
    except RuntimeError as error:
        print(str(error))
        sys.exit(1)

    print_next_steps(project_dir, args.name, tracker_tag)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "new-project":
        run_new_project(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        print(f"tracker {__version__}  ({TRACKER_REPO_URL})")
        return

    from .app import main as app_main
    app_main()


if __name__ == "__main__":
    main()
