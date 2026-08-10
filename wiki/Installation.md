# Installation

Install the tracker once per machine. Projects are created separately —
see [Starting a new project](Starting-a-new-project).

---

## Fresh machine

Requires `curl` and `sudo`:

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

The same command also **updates** an existing installation: it detects the repo
already exists, pulls, and re-runs provisioning. Every step is idempotent, so
it is safe to run any time.

---

## Making the generator double-clickable

Once, after installing, mark the launcher for your platform executable:

```bash
cd ~/tracking
chmod +x new_project.py "New Project.command" "New Project.desktop"
```

- **Linux** — double-click `New Project.desktop`. Some desktop environments ask
  you to confirm the first time ("Trust and Launch").
- **macOS** — double-click `New Project.command`. Gatekeeper may block it the
  first time; right-click → Open to allow it.
- **Any platform** — `python3 new_project.py` always works.

---

## uv — Python environment management

uv manages the tracker's Python dependencies (OpenCV, NumPy, pandas, FastAPI)
in an isolated `.venv/` inside the repo. System Python is untouched, and every
machine gets identical versions from `uv.lock`.

Always run tracker commands with `uv run` from anywhere inside the repo. uv
finds `pyproject.toml` by walking up the directory tree and activates the right
environment automatically — no manual activation.

```bash
cd ~/tracking
uv run tracker --project /path/to/my_experiment
```

After `install.sh` installs uv, **restart your shell** before using `uv run`
directly.

---

## TRex

TRex is a separate conda application, not a Python package — `uv` cannot
install it. `install.sh` handles it, but if you need to do it manually:

```bash
conda create -n trex -c trexing trex
```

Only the **tracking** step needs TRex. Analysis-only work on an existing
project does not — see [Reproducing a project](Reproducing-a-project).

### FFmpeg

Annotated tuning clips are written by OpenCV in a codec browsers cannot play,
then re-encoded to H.264 with FFmpeg. Without FFmpeg the clips still open in
VLC but will not play in the Tune tab. Check and install:

```bash
conda activate trex
which ffmpeg || conda install ffmpeg
```

---

## Machine-local configuration

`pipeline/pipeline.yaml` (gitignored, created by `install.sh`):

```yaml
conda_env: trex              # conda env where TRex is installed
miniforge_dir: ~/miniforge3  # path to Miniforge
```

These defaults are correct for most machines. Edit only if your setup differs.

---

## Verifying the install

```bash
cd ~/tracking
uv run tracker --version
```

Should print the tracker version and repo URL.
