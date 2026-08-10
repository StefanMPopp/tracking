# Insect Video Tracking Pipeline

End-to-end workflow for video-based tracking of insects (ants, flies, crickets)
— from parameter selection through track extraction.

**Tracking engine:** [TRex](https://trex.run) v2.x
**Hardware:** Raspberry Pi 4 + HQ camera module, ~30×20 cm arena, LED panel frame
**Stack:** Python, OpenCV, TRex (conda), uv

📖 **[Full documentation is in the wiki](../../wiki)**

---

## Quick start

**Install** (requires `curl` and `sudo`):

```bash
curl -fsSL https://raw.githubusercontent.com/StefanMPopp/tracking/main/install.sh | bash
```

**Create a project** — double-click `New Project.desktop` (Linux) or
`New Project.command` (macOS), or:

```bash
cd ~/tracking
python3 new_project.py            # small window
uv run tracker new-project --name my_experiment --path ~/projects   # or CLI
```

**Work in a project:**

```bash
cd ~/projects/my_experiment
uv sync --extra tracking
uv run jupyter lab 1_pipeline.ipynb
```

---

## Documentation

| Page | Contents |
|---|---|
| **[Installation](../../wiki/Installation)** | install.sh, uv, TRex, FFmpeg, machine config |
| **[Starting a new project](../../wiki/Starting-a-new-project)** | the generator, version tagging, publishing, linking videos |
| **[Reproducing a project](../../wiki/Reproducing-a-project)** | the two reviewer tiers and how `setup.py` works |
| **[Using the tracker](../../wiki/Using-the-tracker)** | all four tabs, shortcuts, troubleshooting |
| **[Configuration reference](../../wiki/Configuration-reference)** | every `project.yaml` key, resolution order, batches |

---

## How the pieces fit together

**This repo** is the tracker — generic, installed once per machine.

**Project repos** (one per experiment) hold configuration, analysis code and
results, and pin the tracker to a git tag so the exact version used to produce
the data can always be recovered. Projects live wherever you choose, **not**
inside this repo.

---

## Repository layout

```
tracking/
    new_project.py            ← generator window (double-clickable launchers beside it)
    New Project.desktop       ← Linux launcher
    New Project.command       ← macOS launcher
    install.sh                ← installer and updater
    setup.sh                  ← alias for install.sh, from inside the repo
    pyproject.toml            ← dependencies; declares the `tracker` command
    uv.lock                   ← exact pinned versions
    pipeline/
        __init__.py           ← package marker; declares __version__
        cli.py                ← `tracker` entry point (app + new-project)
        app.py                ← browser app: background, masks, parameters, tune
        _scaffold.py          ← project generator; enforces version tagging
        _sweep.py             ← TRex invocation, frame annotation
        _background.py        ← background image computation
        _masks.py             ← mask I/O, TRex serialisation, circle detection
        _resolve.py           ← layered config resolution
        _settings.py          ← .settings file read/patch/write
        _tab_*.py             ← one module per app tab
        masks.py              ← standalone mask editor
        tune.py               ← standalone sweep CLI
        default.settings      ← TRex defaults for all projects
        pipeline.yaml.example ← template for machine-local config
    project_template/
        project.yaml          ← copied into each new project
    wiki/                     ← wiki page sources (push to the wiki repo)
```
