# Using the tracker

The tracker is a browser app with one tab per pipeline step. The video and
project config load once, so switching tabs is instant.

---

## Opening it

### From the pipeline notebook (usual way)

Run the **Tracking** cell in `1_pipeline.ipynb`. It launches the app in the
background and prints the URL, so the notebook stays responsive while you work
in the browser.

```python
tracker_process = subprocess.Popen(
    ["uv", "run", "tracker", "--project", str(PROJECT_DIR)],
    cwd=PROJECT_DIR,
)
```

Press `Q` in the browser when finished. The cell below reports whether it has
exited (`tracker_process.poll()` returns `None` while still running).

### Directly

```bash
uv run tracker --project ~/projects/pain_killers
```

Useful flags:

| Flag | Effect |
|---|---|
| `--video NAME` | Open a specific video instead of the first one found |
| `--mode tuning` | Start in Tuning mode (uses `2_tracking/tuning/`) |
| `--tab masks` | Open straight to a tab: `background`, `masks`, `batches`, `tune` |
| `--port 8001` | Use a different port |

`Q` in the browser or `Ctrl+C` in the terminal quits. When the confirmation
dialog is open, `Enter` confirms.

---

## Top bar

- **Project / Tuning toggle** — switches between real trial videos
  (`2_tracking/`) and dedicated tuning clips (`2_tracking/tuning/`). Each mode
  has its own videos, background images and masks.
- **Video picker** — switch the video that Background and Masks operate on.
- **🔗 Link video** — symlink a video from an external drive into the current
  mode's `1_videos/`.

**Ctrl + scroll** zooms images and videos anywhere in the app; double-click
resets. In the Masks tab the zoom is coordinate-aware, so drawing stays
accurate at any zoom level.

---

## 🖼 Background

Produces `average_{video}.png`, which TGrabs uses to separate animals from the
arena.

Set a **method** (`mean`, `median`, `max`, `min`), how many frames to sample
(**n images**), and optionally a **start/end time** in seconds. Frames are
sampled evenly across that range. If you leave the times blank, the first and
last second are trimmed automatically to avoid camera shake from pressing
record.

`median` is usually the best choice: animals moving across the arena are
averaged out, while the static background survives. Restricting the time range
helps when an animal sits still for a long stretch and would otherwise be
baked into the background.

Click **Generate** to add a candidate to the comparison grid — slot 1 is a
video viewer, and up to 3 candidates sit beside it. Click a candidate to select
it (its parameters fill the sidebar); `Del` removes it from the grid without
touching disk. Any existing background image is loaded automatically and
labelled **pre-existing**.

`Ctrl+S` saves the selected candidate to `2_tracking/2_pv/`, asking before
overwriting. Parameters that differ from the project defaults are recorded
under `video_overrides` for that video.

> A dedicated background recording (`{video}_average.{ext}` in `1_videos/`) is
> used automatically if present — the same controls then sample that file
> instead of the tracking video.

---

## 🎯 Masks

Masks define include or ignore regions. They are applied automatically during
threshold sweeps and tracking runs.

Three ways to create shapes:

- **Perimeter** (`P`) — click vertices one by one; `Enter` or click near the
  first point to close
- **Rectangle** (`R`) / **Oval** (`O`) — click and drag
- **Auto-detect** — set circle parameters in the sidebar and click Detect

In Select mode (`S`): drag a shape to move it, drag a vertex to reshape,
drag the yellow handle to rotate, Shift+drag a corner to resize proportionally,
Ctrl+drag to duplicate. Right-click a vertex to delete it, or a shape body to
delete the shape.

`I` / `G` toggle include / ignore. `Ctrl+S` saves the video's shapes;
`Ctrl+Shift+S` saves them as project defaults.

Masks resolve in three tiers — the first tier with any shapes wins outright,
they are not merged:

1. `{video_name}_include_*.csv` — this video only
2. `{batch_name}_include_*.csv` — the batch this video belongs to
3. `default_include_*.csv` — project-wide

### When auto-detect finds nothing

Click **🩺 Debug detection**. It shows the raw HSV colour mask, the mask after
morphological cleanup, and every contour found — colour-coded green (accepted)
or red (rejected) with its measured area and circularity, plus a table giving
the exact rejection reason.

The usual culprit is **uneven lighting across the circle**: parts of the ring
fall outside the hue/saturation/brightness window, the ring breaks up, and
circularity drops below the 0.5 threshold. If stage 1 shows a patchy ring,
widen **hue tolerance**, raise **max brightness**, or lower saturation.

---

## ⚙️ Parameters

Project-wide settings and batch management.

**Project parameters** — arena width (cm), max individuals, individual prefix,
and an optional video-extension override (normally auto-detected from the
files present).

**Batches** group videos that share settings — typically one day or one
treatment. Select videos on the left (click, Shift+click for a range), then add
them to a new or existing batch. A video belongs to at most one batch.

Each batch card carries a description, confirmed threshold, conversion range,
max individuals, individual prefix, a mask thumbnail, and its member list.

Settings resolve most-specific-first:

```
video_overrides → batch → project level → default.settings
```

---

## 🧪 Tune

Available in **Tuning mode** only (greyed out otherwise).

Enter one or more **threshold values** and optionally an **animal size**
min/max in pixels. If you fill only one size bound, the other is taken from
`default.settings`. The **conversion range** here applies to the tuning clip
only and is never written to project or batch level.

**Run sweep** runs TGrabs + TRex once per threshold and renders an annotated
clip for each. It blocks while running; **Stop** cancels it, terminating the
in-flight TRex process rather than waiting for the current threshold to finish.
`Ctrl+C` in the terminal does the same.

Results appear in a 2×2 grid driven by one shared play button and scrub bar, so
all clips advance together. Click a cell to select it, `Del` clears it (the
file stays on disk), and any clip in the list on the left can be double-clicked
into a slot — including clips from earlier sessions.

Selecting a clip fills the **Selected clip's values** panel by reading that
clip's own `.settings` file, so you always see the values that actually
produced it. From there, save the threshold and size to the project defaults or
to any batch.

---

## Where things are written

```
2_tracking/
    2_pv/               average_{video}.png, .pv files
    3_csv_individual/   TRex per-individual CSVs
    masks/              {video}_ / {batch}_ / default_ polygon CSVs
    tuning/
        {video}_thresh{NNN}.settings    settings used for each sweep run
        {video}_t{NNN}.mp4              annotated comparison clips
        csv_{video}_t{NNN}/             per-threshold track output
```
