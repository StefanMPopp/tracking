# Configuration reference

Two configuration files matter, at different scopes.

| File | Scope | Committed? |
|---|---|---|
| `project.yaml` | one experiment | yes, in the project repo |
| `pipeline/pipeline.yaml` | one machine | no (gitignored) |
| `pipeline/default.settings` | TRex defaults for all projects | yes, in this repo |

---

## Resolution order

Any setting can be specified at several levels. The most specific one wins:

```
video_overrides.<video>  →  batches.<batch>  →  project level  →  default.settings
```

So a project-wide `track_max_individuals: 8` can be overridden to `12` for one
batch, and to `15` for one video inside that batch.

Note that `project.yaml` values **always** override `default.settings` — the
latter only supplies values the project does not mention at all.

---

## `project.yaml`

Lives at the project root (not inside `2_tracking/`), because the pipeline
notebook reads it too.

### Required

| Field | Description |
|---|---|
| `meta_real_width` | Physical arena width in cm |
| `track_max_individuals` | Expected number of individuals |
| `individual_prefix` | ID prefix in TRex output, e.g. `ant` |

All three are editable in the app's **Parameters** tab.

### Optional

| Field | Description |
|---|---|
| `video_extension` | Override the auto-detected extension (normally inferred from the files in `1_videos/`) |
| `videos` | List of video names, without extension |
| `sweep_thresholds` | `detect_threshold` values to compare in the Tune tab |
| `confirmed_detect_threshold` | Written when you save a chosen threshold |
| `animal_size_min` / `animal_size_max` | Blob size filter in px; becomes `detect_size_filter` and `track_size_filter` |
| `video_conversion_range` | `[start_frame, end_frame]` to process |
| `background_params` | Defaults for the Background tab (see below) |
| `auto_detect_circles` | Defaults for mask circle detection (see below) |
| `trail_seconds` | Length of the motion trail in annotated clips |
| `grid_cols` | Layout of the comparison grid |
| `batches` | Named groups of videos (see below) |
| `video_overrides` | Per-video overrides of any key above |

### `background_params`

```yaml
background_params:
  method: median       # mean | median | max | min
  n_images: 10
  start_time: null     # seconds; null = auto (video start + 1s)
  end_time: null       # seconds; null = auto (video end − 1s)
```

### `auto_detect_circles`

```yaml
auto_detect_circles:
  expected_count: 1
  diameter_cm: 4.5
  thickness_cm: 0.3
  hue_center: 0        # OpenCV HSV hue 0–179; 0 = red, 60 = green, 120 = blue
  hue_tolerance: 15
  value_max: 120       # max brightness; low = dark colours
```

### `batches`

Group videos that share settings. A video must not appear in more than one
batch. Masks support a matching tier: `masks/{batch_name}_include_*.csv`.

```yaml
batches:
  day1_control:
    videos: [vid_001, vid_002, vid_003]
    description: "Control group, day 1, 22 °C"
    confirmed_detect_threshold: 50
    video_conversion_range: [0, 900]
  day1_treatment:
    videos: [vid_004, vid_005]
    confirmed_detect_threshold: 65
```

Manage these in the **Parameters** tab rather than by hand where possible.

### `video_overrides`

```yaml
video_overrides:
  vid_003:
    video_conversion_range: [30, 900]
    confirmed_detect_threshold: 55
```

`video_conversion_range` is the most common use, since it genuinely varies per
recording.

---

## `pipeline/pipeline.yaml`

Machine-local, gitignored, created by `install.sh`:

```yaml
conda_env: trex              # conda env where TRex is installed
miniforge_dir: ~/miniforge3  # path to Miniforge
```

Edit only if your setup differs from the defaults.

---

## `pipeline/default.settings`

TRex settings applied to every project unless overridden. Notable keys:

```
detect_threshold = 30
detect_size_filter = [[90,24000]]
track_size_filter = [[90,20000]]
track_max_individuals = 8
video_conversion_range = [0,150]
```

Change these only when the new value should be the baseline for **all**
projects. For a single experiment, set it in that project's `project.yaml`
instead.
