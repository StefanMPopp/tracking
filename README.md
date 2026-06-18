# Insect Video Tracking Pipeline

End-to-end workflow for video-based tracking of insects (ants, flies, crickets, etc.) — from parameter selection through track extraction.

**Tracking engine**: [TRex](https://trex.run) (v2.x)  
**Hardware**: Raspberry Pi 4 + HQ camera module, ~30×20 cm arena, LED panel frame  
**Stack**: Python, OpenCV, TRex (conda), uv

---

## Installation

### On a fresh machine

Download and run the installer (requires `curl` and `sudo`):

```bash
curl -fsSL https://raw.githubusercontent.com/StefanMPopp/tracking/main/install.sh | bash
```

This will:
1. Install `git` (if absent)
2. Clone this repo to `~/tracking`
3. Install Miniforge (if absent)
4. Create a `trex` conda environment and install TRex
5. Install `uv` (if absent) — **restart your shell after this step if running manually**
6. Sync Python dependencies via `uv sync`
7. Create `pipeline/pipeline.yaml` from the template

The same command updates an existing installation — it detects the repo already exists, pulls the latest changes, and re-runs all provisioning steps. Every step is idempotent, so it is safe to run at any time.

---

## uv — Python environment management

uv manages the pipeline's Python dependencies (OpenCV, NumPy, pandas, etc.) in an isolated virtual environment (`.venv/`) inside the repo. This keeps system Python untouched and ensures every machine runs the exact same package versions, as defined in `uv.lock`.

**How to use it**: always run pipeline scripts with `uv run` from anywhere inside the repo root or below. uv finds `pyproject.toml` by walking up the directory tree and activates the correct environment automatically — you never need to activate it manually.

```bash
cd ~/tracking
uv run python pipeline/tune.py --project /path/to/project --video my_video
```

After `install.sh` installs uv, **restart your shell** before using `uv run` directly (the installer does this automatically as part of the full install sequence).

---

## Starting a new project

Projects live **outside** the repo — they contain large video and data files that should not be version-controlled alongside the pipeline code.

```bash
cd ~/tracking
uv run python pipeline/new_project.py --name pain_killers --path /path/to/projects
```

This creates the full folder and file scaffold:

```
/path/to/projects/pain_killers/
    project.yaml          ← edit this before doing anything else
    1_videos/             ← place input videos here
    2_pv/                 ← TGrabs output (.pv files, background images)
    3_csv_individual/     ← TRex per-individual CSV output
    4_csv_video/          ← collated per-video CSV (later pipeline stage)
    tuning/               ← threshold sweep outputs
    masks/                ← include/ignore polygon CSVs for TRex
```

Edit `project.yaml` to set at minimum `meta_real_width`, `track_max_individuals`, and `videos`.

---

## Background image (optional but recommended)

Record a short video of the empty arena before the experiment, named `{video_name}_average.{video_extension}`, and place it in `1_videos/`. The pipeline extracts the middle frame and saves it where TGrabs expects it. If no background video is provided, TGrabs computes the background automatically from the tracking video.

---

## Creating masks (optional)

Masks define include or ignore regions for TRex tracking. They are automatically
applied during threshold sweeps and full tracking runs.

```bash
uv run python pipeline/masks.py \
    --project /path/to/projects/pain_killers \
    --video pain_test
```

Opens a browser-based editor. Three ways to create shapes:

- **Perimeter** (`P`): click vertices one by one, press Enter or click near the first point to close
- **Rectangle** (`R`) / **Oval** (`O`): click and drag
- **Auto-detect**: set circle parameters in the sidebar and click Detect

In Select mode (`S`): drag a shape to move it, drag a vertex to reshape it, drag
the yellow rotation handle to rotate it. Right-click a vertex to delete it;
right-click a shape body to delete the shape.

Toggle mask type with `I` (include) / `G` (ignore). Press **Save all unsaved shapes**
when done. Masks are saved to `masks/{video_name}_include_1.csv` etc.

**Project defaults**: shapes shared across videos. Load with "Load project defaults"
(adds to current canvas); save with "Save current as defaults".

After creating masks, re-run `tune.py` — masks are picked up automatically.

---



Before tracking, find the optimal `detect_threshold` for your recording conditions:

```bash
uv run python pipeline/tune.py \
    --project /path/to/projects/pain_killers \
    --video pain_test
```

Override the threshold values to test (overrides `project.yaml`):

```bash
uv run python pipeline/tune.py \
    --project /path/to/projects/pain_killers \
    --video pain_test \
    --thresholds 15,25,35,50
```

The script runs TGrabs + TRex for each threshold value, renders an annotated clip per threshold (showing blob area in px/cm² and centroid speed), composes a side-by-side grid video, opens it, and prompts you to accept a value or re-sweep. The confirmed threshold is written back to `project.yaml`.

Sweep outputs are saved under `tuning/sweep_{timestamp}/`. A cumulative log of all sweeps and confirmed values is kept in `tuning/tuning_log.json`.

---

## Repository layout

```
tracking/
    pipeline/
        tune.py                  ← threshold sweep CLI
        new_project.py           ← creates folder and file scaffold for a new project (4_csv_video, tuning, etc.)
        _sweep.py                ← TRex invocation, frame annotation, grid compositor
        _background.py           ← background image extraction and copying
        _settings.py             ← .settings file read/patch/write utilities
        default.settings         ← shared TRex default settings (edit per project via project.yaml)
        pipeline.yaml            ← machine-local config: conda env name, Miniforge path (gitignored)
        pipeline.yaml.example    ← committed template for pipeline.yaml
    project_template/
        project.yaml             ← blank template with all keys documented; copied by new_project.py
    install.sh                   ← installer and updater: clones or pulls, then provisions
    setup.sh                     ← thin alias for install.sh; use from inside the repo
    pyproject.toml               ← declares Python dependencies; used by uv to build the environment
    uv.lock                      ← exact pinned versions of all dependencies; committed to the repo
    .gitignore
    README.md
```

---

## Machine-local configuration

`pipeline/pipeline.yaml` (gitignored, created automatically by `install.sh`):

```yaml
conda_env: trex              # name of the conda env where TRex is installed
miniforge_dir: ~/miniforge3  # path to Miniforge
```

These defaults are correct for all machines including the dev machine. Edit only if your setup differs.

---

## Per-project configuration

`project.yaml` lives with the project data, not in this repo. Key fields:

| Field | Description |
|---|---|
| `meta_real_width` | Physical arena width in cm |
| `track_max_individuals` | Expected number of individuals |
| `individual_prefix` | ID prefix in TRex output (e.g. `ant`) |
| `video_extension` | File extension for all videos in the project (default: `MP4`) |
| `videos` | List of video names (without extension) |
| `sweep_thresholds` | List of `detect_threshold` values to compare in `tune.py` |
| `confirmed_detect_threshold` | Set automatically by `tune.py` after sweep review |
| `video_overrides` | Per-video overrides for any project-level key; `video_conversion_range` goes here |
