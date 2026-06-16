# Insect Video Tracking Pipeline

PAT: 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

End-to-end workflow for video-based tracking of insects (ants, flies, crickets, etc.) — from parameter selection through track extraction.

**Tracking engine**: [TRex](https://trex.run) (v2.x)  
**Hardware**: Raspberry Pi 4 + HQ camera module, ~30×20 cm arena, LED panel frame  
**Stack**: Python, OpenCV, TRex (conda), uv

---

## Installation

### On a fresh machine

Download and run the installer (requires `curl` and `sudo`):

```bash
curl -fsSL https://raw.githubusercontent.com/<yourhandle>/tracking/main/install.sh | bash
```

This will:
1. Install `git` (if absent)
2. Clone this repo to `~/tracking`
3. Install Miniforge (if absent)
4. Create a `trex` conda environment and install TRex
5. Install `uv` (if absent) and sync Python dependencies
6. Create `pipeline/pipeline.yaml` from the template

### On Stefan's dev machine (repo already present)

```bash
cd /home/stefan/tracking
bash setup.sh
cp pipeline/pipeline.yaml.example pipeline/pipeline.yaml
```

---

## Starting a new project

Projects live **outside** the repo (they contain large video and data files).

```bash
uv run python pipeline/new_project.py --name pain_killers --path /path/to/projects
```

This creates:
```
/path/to/projects/pain_killers/
    project.yaml          ← edit this before doing anything else
    1_videos/             ← place input videos here
    2_pv/                 ← TGrabs output (.pv files, background images)
    3_csv_individual/     ← TRex per-individual CSV output
    4_csv_trial/          ← collated per-trial CSV (later pipeline stage)
    tuning/               ← threshold sweep outputs
```

Edit `project.yaml` to set `meta_real_width`, `track_max_individuals`, `trials`, etc.

---

## Background image (optional but recommended)

Record a short video of the empty arena before the experiment, named `{trial}_average.MP4`, and place it in `1_videos/`. The pipeline will extract its middle frame and use it as the background for TGrabs. If no background video is provided, TGrabs computes the background automatically from the tracking video.

---

## Threshold sweep

Before tracking, find the optimal `detect_threshold` for your recording conditions:

```bash
uv run python pipeline/tune.py \
    --project /path/to/projects/pain_killers \
    --trial pain_test
```

Optional: override the threshold values to test:

```bash
uv run python pipeline/tune.py \
    --project /path/to/projects/pain_killers \
    --trial pain_test \
    --thresholds 15,25,35,50
```

The script runs TGrabs + TRex for each threshold value, renders an annotated clip per threshold (showing blob area in px/cm² and centroid speed), composes a side-by-side grid video, opens it, and prompts you to accept a value or iterate. The confirmed threshold is written back to `project.yaml`.

Sweep outputs are saved under `tuning/sweep_{timestamp}/`. A cumulative log of all sweeps and confirmed values is kept in `tuning/tuning_log.json`.

---

## Repository layout

```
tracking/
    pipeline/
        tune.py                  ← threshold sweep CLI
        new_project.py           ← project scaffolding
        _sweep.py                ← TRex invocation, annotation, grid compositor
        _background.py           ← background image helpers
        _settings.py             ← .settings file read/patch/write
        default.settings         ← shared TRex default settings
        pipeline.yaml            ← machine-local config (gitignored)
        pipeline.yaml.example    ← template for pipeline.yaml
    project_template/
        project.yaml             ← blank template used by new_project.py
    install.sh                   ← one-shot installer for fresh machines
    setup.sh                     ← internal provisioner (called by install.sh)
    pyproject.toml
    uv.lock
    .gitignore
    README.md
```

---

## Machine-local configuration

`pipeline/pipeline.yaml` (gitignored, created by `setup.sh`):

```yaml
conda_env: trex              # conda env name where TRex is installed
miniforge_dir: ~/miniforge3  # path to Miniforge
```

---

## Per-project configuration

`project.yaml` (lives with the project data, not in this repo):

Key fields:

| Field | Description |
|---|---|
| `meta_real_width` | Physical arena width in cm |
| `track_max_individuals` | Expected number of individuals |
| `individual_prefix` | ID prefix in TRex output (e.g. `ant`) |
| `video_conversion_range` | `[start_frame, end_frame]` for conversion and sweeps |
| `sweep_thresholds` | List of `detect_threshold` values to compare |
| `confirmed_detect_threshold` | Set by `tune.py` after sweep review |
| `trials` | List of trial names (without extension) |
| `trial_overrides` | Per-trial overrides for any project-level key |
