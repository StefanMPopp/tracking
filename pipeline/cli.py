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
    bump_and_tag,
    commit_and_push,
    next_patch_version,
    print_next_steps,
    resolve_tracker_tag,
    scaffold_project,
)

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT   = PACKAGE_DIR.parent


def run_push(argv: list[str]) -> None:
    """
    Commit and push everyday tracker edits in one step. Separate from
    `release` on purpose — this runs constantly during development and
    should never require thinking about version numbers; `release` runs
    occasionally and is only ever about tagging.
    """
    parser = argparse.ArgumentParser(
        prog="tracker push",
        description="Commit all changes and push, in one step.",
    )
    parser.add_argument("message", help="Commit message")
    parser.add_argument(
        "--release", action="store_true",
        help="Also tag a release immediately afterward (runs `tracker release`)",
    )
    args = parser.parse_args(argv)

    try:
        pushed = commit_and_push(REPO_ROOT, args.message)
    except RuntimeError as error:
        print(str(error))
        sys.exit(1)

    print("Nothing to commit or push." if not pushed else "Committed and pushed.")

    if args.release:
        print("")
        run_release([])


def run_release(argv: list[str]) -> None:
    """
    Tag the tracker at its current commit, so future `new-project` runs pin
    to it. A deliberate, separate step from creating a project — run this
    when tracker changes are finished and pushed, not when you happen to
    want to start a project.
    """
    parser = argparse.ArgumentParser(
        prog="tracker release",
        description="Tag this tracker version so new projects can pin to it.",
    )
    parser.add_argument(
        "--version", default=None,
        help="New version, e.g. 0.4.0 (default: next patch version)",
    )
    args = parser.parse_args(argv)

    new_version = args.version or next_patch_version(__version__)
    tag = f"v{new_version}"

    print(f"Current version: {__version__}")
    print(f"New version:     {new_version}  (tag {tag})")
    answer = input("Bump, commit, tag, and push? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    try:
        bump_and_tag(REPO_ROOT, new_version)
    except RuntimeError as error:
        print("")
        print(str(error))
        sys.exit(1)

    print("")
    print(f"Tagged and pushed {tag}.")
    print("New projects created from now on will pin to this version.")


def run_new_project(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="tracker new-project",
        description="Scaffold a new experiment project that uses this tracker.",
    )
    parser.add_argument("--name", required=True,
                        help="Project name, e.g. pain_killers")
    parser.add_argument("--path", default=str(Path.home() / "Documents"),
                        help="Parent directory for the new project folder "
                             "(default: ~/Documents)")
    args = parser.parse_args(argv)

    try:
        tracker_tag = resolve_tracker_tag(REPO_ROOT, __version__)
    except RuntimeError as error:
        print("")
        print("Cannot create a project — the tracker itself needs releasing first:")
        print("")
        print(str(error))
        print("")
        print("Run:  uv run tracker release")
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

    if len(sys.argv) > 1 and sys.argv[1] == "release":
        run_release(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "push":
        run_push(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        print(f"tracker {__version__}  ({TRACKER_REPO_URL})")
        return

    from .app import main as app_main
    app_main()


if __name__ == "__main__":
    main()
