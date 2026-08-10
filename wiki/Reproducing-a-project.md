# Reproducing a project

This is the reviewer's view: what someone does when they clone one of your
project repos. It is also what you do when moving a project to a new machine.

Every generated project ships with its own `setup.py` and a README containing
these instructions, so a reviewer never needs to read the tracker's
documentation.

---

## Two tiers

Choose based on what you need to reproduce.

### Analysis only

Uses the tracking data already committed to the project repo. No TRex, no raw
videos, no conda.

**Requires:** Python 3.11+

```bash
curl -fsSL https://raw.githubusercontent.com/StefanMPopp/pain_killers/main/setup.py -o setup.py
python3 setup.py
```

### Full reproduction

Also re-runs the tracking step from raw video.

**Requires:** Python 3.11+, plus [TRex](https://trex.run/docs/install.html) in
a conda environment (installed separately — it is not a Python package).

```bash
curl -fsSL https://raw.githubusercontent.com/StefanMPopp/pain_killers/main/setup.py -o setup.py
python3 setup.py --with-tracking
```

You will be asked where the raw videos are, and they get symlinked into
`2_tracking/1_videos/`. Leave the prompt blank to skip — the committed tracking
outputs are then used as-is.

---

## What `setup.py` does

1. Checks for `uv`, offering to install it if missing
2. Clones the project repo if the script was downloaded standalone
   (skipped when run from inside an existing clone)
3. `uv sync` — analysis dependencies, or `uv sync --extra tracking` for the
   full tier, which also pulls the tracker at its pinned tag
4. For `--with-tracking`: asks for the raw video folder and creates symlinks
5. Prints the next command to run

It is safe to re-run at any time.

---

## Running the workflow

```bash
uv run jupyter lab 1_pipeline.ipynb
```

Run the notebook top to bottom:

1. **Setup** — loads `project.yaml`, defines paths
2. **Preprocessing** — experiment-specific video preparation
3. **Tracking** — launches the tracker app in your browser. This step is
   interactive; see [Using the tracker](Using-the-tracker)
4. **Postprocessing** — collating per-individual tracks into analysis-ready data

Experiment-specific analysis lives in `3_analysis/`.

---

## Why the tracker version is pinned

The project's `pyproject.toml` pins an exact tag:

```toml
[project.optional-dependencies]
tracking = [
    "tracking @ git+https://github.com/StefanMPopp/tracking@v0.3.0",
]
```

Tracker changes can alter tracking output. Pinning means re-running the
pipeline years later reproduces the original results rather than silently
picking up newer behaviour.

If you deliberately change the pin, re-run the **whole** tracking step rather
than mixing outputs from two tracker versions in one dataset.
