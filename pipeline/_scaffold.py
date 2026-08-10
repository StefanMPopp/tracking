"""
_scaffold.py — generate a new project repo that uses this tracker.

Run via:  tracker new-project --name my_experiment --path ~/projects

The generated project is a self-contained, shareable GitHub repo. It pins
this tracker to the git tag currently checked out, so a reviewer cloning
the project years later gets the exact tracker version used to produce
the data.

Because reproducibility depends on that pin, scaffolding refuses to run
from an untagged tracker checkout and tells you which tag to create.
"""

import json
import subprocess
from pathlib import Path

TRACKER_REPO_URL = "https://github.com/StefanMPopp/tracking"

PROJECT_SUBDIRS = [
    "2_tracking/1_videos",
    "2_tracking/2_pv",
    "2_tracking/3_csv_individual",
    "2_tracking/4_csv_video",
    "2_tracking/masks",
    "2_tracking/tuning/1_videos",
    "2_tracking/tuning/2_pv",
    "2_tracking/tuning/masks",
    "3_analysis",
]


def next_patch_version(version: str) -> str:
    """'0.3.0' -> '0.3.1'. Falls back to appending '.1' if unparseable."""
    parts = version.split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        major, minor, patch = parts
        return f"{major}.{minor}.{int(patch) + 1}"
    return f"{version}.1"


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def commit_and_push(repo_root: Path, message: str) -> bool:
    """
    Stage everything, commit with the given message, and push the current
    branch. Returns False (no-op) if there was nothing to commit and nothing
    to push; True if a commit and/or push actually happened.

    This is the everyday "ship my edits" action — separate from
    bump_and_tag(), which is only ever about tagging a version.
    """
    committed = False
    if _git(repo_root, "status", "--porcelain", check=False):
        _git(repo_root, "add", "-A")
        _git(repo_root, "commit", "-m", message)
        committed = True

    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    ahead = _git(
        repo_root, "rev-list", f"origin/{branch}..HEAD", "--count", check=False
    )
    if not committed and (not ahead or ahead == "0"):
        return False

    _git(repo_root, "push", "origin", branch)
    return True


def bump_and_tag(repo_root: Path, new_version: str) -> str:
    """
    Set new_version in pyproject.toml and pipeline/__init__.py, commit that
    change (only if something actually changed), tag it, and push both the
    branch and the tag. Returns the tag.

    Refuses to run if the tree has unrelated uncommitted changes, so the
    version-bump commit doesn't silently sweep up other in-progress work —
    commit or use commit_and_push() first.
    """
    import re
    import shutil

    def git(*args: str, check: bool = True) -> str:
        return _git(repo_root, *args, check=check)

    tag = f"v{new_version}"

    if git("tag", "-l", tag, check=False):
        raise RuntimeError(
            f"Tag '{tag}' already exists. Choose a version that hasn't been used."
        )

    dirty = git("status", "--porcelain", check=False)
    if dirty:
        raise RuntimeError(
            "The tracker repo has uncommitted changes:\n\n"
            f"{dirty}\n\n"
            "Commit or stash them first, so the version-bump commit only "
            "contains the version change."
        )

    pyproject_file = repo_root / "pyproject.toml"
    init_file       = repo_root / "pipeline" / "__init__.py"

    pyproject_text, pyproject_count = re.subn(
        r'(?m)^version = "[^"]*"', f'version = "{new_version}"',
        pyproject_file.read_text(), count=1,
    )
    init_text, init_count = re.subn(
        r'(?m)^__version__ = "[^"]*"', f'__version__ = "{new_version}"',
        init_file.read_text(), count=1,
    )
    if pyproject_count == 0 or init_count == 0:
        raise RuntimeError(
            "Could not find the version line in pyproject.toml or "
            "pipeline/__init__.py to update."
        )

    pyproject_file.write_text(pyproject_text)
    init_file.write_text(init_text)

    lock_file = repo_root / "uv.lock"
    uv_path   = shutil.which("uv")
    if uv_path:
        subprocess.run([uv_path, "lock"], cwd=repo_root, capture_output=True, text=True)
        # A version-only change to the local project can leave uv.lock
        # unchanged; only stage it if `uv lock` actually touched it.

    changed_files = [pyproject_file, init_file]
    if lock_file.exists():
        changed_files.append(lock_file)

    if git("status", "--porcelain", check=False):
        git("add", *[str(f) for f in changed_files])
        git("commit", "-m", f"Bump version to {new_version}")

    git("tag", tag)
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    git("push", "origin", branch)
    git("push", "origin", tag)

    return tag


# =============================================================================
# Tag resolution
# =============================================================================

def resolve_tracker_tag(tracker_repo_dir: Path, declared_version: str) -> str:
    """
    Return the git tag checked out in the tracker repo.

    Raises RuntimeError with actionable instructions if HEAD is untagged or
    if the tag disagrees with the version in pyproject.toml — either case
    means a project pinned now would not be reproducible.
    """
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(tracker_repo_dir), *args],
            capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    if not git("rev-parse", "--git-dir"):
        raise RuntimeError(
            f"{tracker_repo_dir} is not a git repository, so the tracker "
            f"version cannot be pinned.\n"
            f"Install the tracker by cloning {TRACKER_REPO_URL} rather than "
            f"copying the files."
        )

    tag = git("describe", "--tags", "--exact-match", "HEAD")
    expected_tag = f"v{declared_version}"

    if not tag:
        raise RuntimeError(
            "The tracker's current commit is not tagged, so a new project "
            "cannot pin a reproducible version.\n\n"
            f"  Version in pyproject.toml:  {declared_version}\n\n"
            "Create and push the tag, then re-run this command:\n\n"
            f"  git -C {tracker_repo_dir} tag {expected_tag}\n"
            f"  git -C {tracker_repo_dir} push origin {expected_tag}\n"
        )

    if tag != expected_tag:
        raise RuntimeError(
            f"Tracker tag '{tag}' does not match the version declared in "
            f"pyproject.toml ('{declared_version}', so the tag should be "
            f"'{expected_tag}').\n\n"
            "Fix one of them so they agree, then re-run this command."
        )

    if git("status", "--porcelain"):
        raise RuntimeError(
            f"The tracker repo has uncommitted changes, so tag '{tag}' does "
            f"not describe the code that would actually run.\n"
            "Commit or stash them, then re-run this command."
        )

    return tag


# =============================================================================
# File templates
# =============================================================================

def _setup_py(project_name: str, tracker_tag: str) -> str:
    return f'''"""
setup.py — reconstruct this project's environment.

Two tiers:

    python3 setup.py                  analysis only (no TRex needed)
    python3 setup.py --with-tracking   also install the tracker and link videos

Safe to re-run at any time.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_NAME = "{project_name}"
REPO_URL     = "https://github.com/StefanMPopp/{project_name}"
TRACKER_TAG  = "{tracker_tag}"


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {{' '.join(command)}}")
    return subprocess.run(command, check=True, **kwargs)


def ensure_uv() -> None:
    """uv manages the Python environment; install it if missing."""
    if shutil.which("uv"):
        return
    print("uv is not installed. It is required to set up this project.")
    answer = input("Install uv now? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        print("Aborted. Install uv manually: https://docs.astral.sh/uv/")
        sys.exit(1)
    run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    if not shutil.which("uv"):
        print("uv was installed but is not on PATH yet.")
        print("Open a new terminal (or source your shell profile) and re-run this script.")
        sys.exit(1)


def ensure_repo() -> Path:
    """
    Return the project directory, cloning the repo first if this script was
    downloaded standalone rather than run from inside a clone.
    """
    here = Path(__file__).resolve().parent
    if (here / "pyproject.toml").exists():
        return here

    target = Path.cwd() / PROJECT_NAME
    if not target.exists():
        print(f"Cloning {{REPO_URL}} …")
        run(["git", "clone", REPO_URL, str(target)])
    return target


def link_videos(project_dir: Path) -> None:
    """
    Ask where the raw videos are and symlink them into 2_tracking/1_videos/.
    Videos are usually too large to commit, so they live outside the repo.
    """
    videos_dir = project_dir / "2_tracking" / "1_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("The raw videos are not part of this repository.")
    print("If you have them, give the folder containing them to link them in.")
    print("Leave blank to skip — tracking outputs already in the repo will be used.")
    source_input = input("Path to videos folder (blank to skip): ").strip()
    if not source_input:
        print("Skipped video linking.")
        return

    source_dir = Path(source_input).expanduser().resolve()
    if not source_dir.is_dir():
        print(f"Not a folder: {{source_dir}} — skipping.")
        return

    video_suffixes = {{".mp4", ".MP4", ".mov", ".MOV", ".avi", ".mkv"}}
    linked = 0
    for video_file in sorted(source_dir.iterdir()):
        if video_file.suffix not in video_suffixes:
            continue
        link_path = videos_dir / video_file.name
        if link_path.exists() or link_path.is_symlink():
            continue
        link_path.symlink_to(video_file)
        linked += 1
    print(f"Linked {{linked}} video(s) into {{videos_dir}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Set up {{PROJECT_NAME}}.")
    parser.add_argument(
        "--with-tracking", action="store_true",
        help="Also install the tracker and link raw videos (needs TRex separately).",
    )
    args = parser.parse_args()

    ensure_uv()
    project_dir = ensure_repo()
    print(f"Project directory: {{project_dir}}")

    print("")
    print("Installing dependencies …")
    if args.with_tracking:
        run(["uv", "sync", "--extra", "tracking"], cwd=project_dir)
    else:
        run(["uv", "sync"], cwd=project_dir)

    if args.with_tracking:
        link_videos(project_dir)

    print("")
    print("=" * 60)
    print("Setup complete.")
    print("=" * 60)
    print("")
    print("Open the pipeline notebook to run the workflow:")
    print(f"  cd {{project_dir}}")
    print("  uv run jupyter lab 1_pipeline.ipynb")
    if args.with_tracking:
        print("")
        print(f"Tracking requires TRex ({{TRACKER_TAG}} pins the tracker itself,")
        print("but TRex is a separate conda install):")
        print("  https://trex.run/docs/install.html")
    else:
        print("")
        print("Installed in analysis-only mode. To also reproduce the tracking step:")
        print("  python3 setup.py --with-tracking")


if __name__ == "__main__":
    main()
'''


def _pyproject_toml(project_name: str, tracker_tag: str) -> str:
    package_name = project_name.replace("_", "-")
    return f'''[project]
name = "{package_name}"
version = "0.1.0"
description = "Experiment: {project_name}"
requires-python = ">=3.11"

# Analysis dependencies — always installed.
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyterlab>=4.0",
    "pyyaml>=6.0",
]

# Tracking dependencies — installed only with:  uv sync --extra tracking
# The tracker is pinned to a tag so this project stays reproducible.
[project.optional-dependencies]
tracking = [
    "tracking @ git+{TRACKER_REPO_URL}@{tracker_tag}",
]

[tool.uv]
package = false
'''


def _readme(project_name: str, tracker_tag: str) -> str:
    return f'''# {project_name}

<!-- One-line description of the experiment. -->

---

## Reproducing this project

Pick the tier you need. Both are safe to re-run.

### Analysis only

Uses the tracking data already committed to this repo. No TRex, no raw videos.

**Requires:** Python 3.11+

```bash
curl -fsSL https://raw.githubusercontent.com/StefanMPopp/{project_name}/main/setup.py -o setup.py
python3 setup.py
```

### Full reproduction (raw video → tracks → analysis)

Also re-runs the tracking step.

**Requires:** Python 3.11+, plus [TRex](https://trex.run/docs/install.html)
in a conda environment (installed separately — it is not a Python package).

```bash
curl -fsSL https://raw.githubusercontent.com/StefanMPopp/{project_name}/main/setup.py -o setup.py
python3 setup.py --with-tracking
```

You will be asked where the raw videos are. They are not in this repo
(too large); leave the prompt blank to skip and reuse the committed
tracking outputs.

---

## Running the workflow

```bash
uv run jupyter lab 1_pipeline.ipynb
```

Run the notebook top to bottom. It covers, in order:

1. **Preprocessing** — preparing raw videos
2. **Tracking** — launches the tracker app in your browser. This step is
   interactive: work through the tabs, then press `Q` in the browser (or
   close the tab) to return to the notebook.
3. **Postprocessing** — collating per-individual tracks into analysis-ready data

Project-specific analysis lives in `3_analysis/`.

---

## Project layout

```
{project_name}/
    setup.py            environment reconstruction (run this first)
    pyproject.toml      dependencies; pins the tracker version
    uv.lock             exact resolved versions (committed)
    project.yaml        tracker configuration
    1_pipeline.ipynb    preprocessing -> tracking -> postprocessing
    2_tracking/         tracker working directory
        1_videos/       symlinks to raw videos (not committed)
        2_pv/           TRex intermediate format
        3_csv_individual/   per-individual tracks
        4_csv_video/    per-video collated tracks
        masks/          arena masks
        tuning/         threshold tuning workspace
    3_analysis/         project-specific analysis
```

---

## Tracker version

This project is pinned to tracker **{tracker_tag}**:
{TRACKER_REPO_URL}/releases/tag/{tracker_tag}

Changing the pin means the tracking step may produce different results.
If you change it, re-run the tracking step rather than mixing outputs.
'''


def _gitignore() -> str:
    return '''# Raw videos — linked in from outside the repo, never committed
2_tracking/1_videos/*
!2_tracking/1_videos/.gitkeep
2_tracking/tuning/1_videos/*
!2_tracking/tuning/1_videos/.gitkeep

# Large intermediates — regenerable from the videos
2_tracking/2_pv/
2_tracking/tuning/2_pv/
2_tracking/tuning/*.mp4
2_tracking/tuning/csv_*/

# Python
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
'''


def _pipeline_notebook(project_name: str) -> str:
    """
    The pipeline notebook. Tracking launches in the background (Popen) so the
    cell returns immediately and the notebook stays responsive while you work
    in the browser.
    """
    def code(source: str) -> dict:
        return {
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": source.strip().split("\n"),
        }

    def markdown(source: str) -> dict:
        return {
            "cell_type": "markdown", "metadata": {},
            "source": source.strip().split("\n"),
        }

    cells = [
        markdown(f"""
# {project_name} — pipeline

Run top to bottom. Sections:

1. Setup
2. Preprocessing
3. Tracking (interactive)
4. Postprocessing
        """),
        markdown("## 1. Setup"),
        code("""
from pathlib import Path
import subprocess
import yaml

PROJECT_DIR  = Path.cwd()
TRACKING_DIR = PROJECT_DIR / "2_tracking"
ANALYSIS_DIR = PROJECT_DIR / "3_analysis"

project_config = yaml.safe_load((PROJECT_DIR / "project.yaml").read_text())
project_config
        """),
        markdown("""
## 2. Preprocessing

Project-specific video preparation — trimming, renaming, quality checks.
Add cells as needed.
        """),
        code("""
# Preprocessing goes here.
        """),
        markdown("""
## 3. Tracking

Launches the tracker app in your browser. It starts in the background, so
this cell returns immediately and the notebook stays usable.

Work through the tabs (Background -> Masks -> Tune -> Parameters), then
press `Q` in the browser or close the tab when finished.
        """),
        code("""
tracker_process = subprocess.Popen(
    ["uv", "run", "tracker", "--project", str(PROJECT_DIR)],
    cwd=PROJECT_DIR,
)
print(f"Tracker running (pid {tracker_process.pid}) at http://localhost:8000")
print("Press Q in the browser when finished, then continue below.")
        """),
        code("""
# Optional: confirm the tracker has exited before continuing.
# returncode is None while it is still running.
tracker_process.poll()
        """),
        markdown("""
## 4. Postprocessing

Collate per-individual tracks into analysis-ready data.
        """),
        code("""
individual_csv_dir = TRACKING_DIR / "3_csv_individual"
sorted(individual_csv_dir.glob("*.csv"))[:10]
        """),
        code("""
# Postprocessing goes here.
        """),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1)


# =============================================================================
# Scaffolding
# =============================================================================

def scaffold_project(
    project_name: str,
    projects_dir: Path,
    tracker_tag: str,
    project_template_file: Path,
) -> Path:
    """Create the project folder, files, and subdirectories. Returns its path."""
    project_dir = projects_dir / project_name
    if project_dir.exists():
        raise RuntimeError(
            f"Project folder already exists: {project_dir}\n"
            "Delete it or choose a different --name."
        )

    project_dir.mkdir(parents=True)
    for subdir in PROJECT_SUBDIRS:
        (project_dir / subdir).mkdir(parents=True)

    # Keep otherwise-empty, gitignored video folders present in the repo
    for keep_dir in ["2_tracking/1_videos", "2_tracking/tuning/1_videos"]:
        (project_dir / keep_dir / ".gitkeep").touch()

    (project_dir / "setup.py").write_text(_setup_py(project_name, tracker_tag))
    (project_dir / "pyproject.toml").write_text(_pyproject_toml(project_name, tracker_tag))
    (project_dir / "README.md").write_text(_readme(project_name, tracker_tag))
    (project_dir / ".gitignore").write_text(_gitignore())
    (project_dir / "1_pipeline.ipynb").write_text(_pipeline_notebook(project_name))

    project_yaml_text = project_template_file.read_text().replace(
        "__PROJECT_NAME__", project_name
    )
    (project_dir / "project.yaml").write_text(project_yaml_text)

    return project_dir


def print_next_steps(project_dir: Path, project_name: str, tracker_tag: str) -> None:
    print("")
    print("=" * 64)
    print(f"Project created: {project_dir}")
    print(f"Tracker pinned to: {tracker_tag}")
    print("=" * 64)
    print("")
    print("Next steps:")
    print("")
    print(f"  1. Fill in the required fields in {project_dir / 'project.yaml'}")
    print("       meta_real_width, track_max_individuals, individual_prefix")
    print("     (or leave them and set them in the app's Parameters tab)")
    print("")
    print("  2. Publish the repo so it can be shared and reproduced:")
    print(f"       cd {project_dir}")
    print("       git init && git add -A && git commit -m 'Initial scaffold'")
    print(f"       gh repo create StefanMPopp/{project_name} --private --source=. --push")
    print("")
    print("  3. Start working:")
    print("       uv sync --extra tracking")
    print("       uv run jupyter lab 1_pipeline.ipynb")
