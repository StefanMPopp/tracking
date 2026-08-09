"""
_tab_tune.py — Tuning tab for the unified pipeline app.

Only meaningful in tuning mode (state["mode"] == "tuning"); the tab is
greyed out / non-interactive in project mode (enforced client-side, since
the underlying routes are otherwise mode-agnostic and harmless to call).

Lets the user:
  - Set sweep parameters: threshold (multiple values), animal size (min/max)
  - video_conversion_range is tuning-clip-specific only (never written to
    project-level or batch-level config — it's edited here directly and
    saved only as a video_overrides entry for this tuning video)
  - Run the sweep (blocking call; reuses run_sweep from _sweep.py unchanged)
  - View up to 4 resulting annotated clips in a 2x2 grid with synchronized
    playback (one shared play/pause + scrub bar driving all four <video>
    elements)
  - Select a grid slot, clear it (memory only — does not delete the file),
    and replace it with any existing {video}_t*.mp4 clip in the tuning
    folder via a double-click-to-select list
"""

import logging
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from _resolve import resolve_batch_for_video, resolve_effective_config
from _settings import read_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic models
# =============================================================================

class RunSweepRequest(BaseModel):
    thresholds:      list[int]
    animal_size_min: int | None = None
    animal_size_max: int | None = None

class UpdateConversionRangeRequest(BaseModel):
    video_conversion_range: list[int] | None = None   # None clears the override

class SaveValuesRequest(BaseModel):
    confirmed_detect_threshold: int | None = None
    animal_size_min:            int | None = None
    animal_size_max:             int | None = None

class SaveToBatchRequest(SaveValuesRequest):
    batch_name: str


# =============================================================================
# Route registration
# =============================================================================

def register_tune_routes(app: FastAPI, state: dict) -> None:

    @app.get("/tuning/config")
    def get_config():
        """
        Effective config for the current tuning video (threshold/animal-size
        defaults if previously set, video_conversion_range, etc.) plus the
        list of existing annotated clips already in the tuning folder.
        """
        project_config = _read_project_config(state)
        video_name     = state["video_name"]

        effective_config = resolve_effective_config(video_name, project_config) \
            if video_name else {}

        default_settings = read_settings(state["base_settings_file"])
        default_thresholds = effective_config.get("sweep_thresholds", [40, 50, 60])

        return JSONResponse({
            "video_name":             video_name,
            "sweep_thresholds":       default_thresholds,
            "animal_size_min":        effective_config.get("animal_size_min"),
            "animal_size_max":        effective_config.get("animal_size_max"),
            "video_conversion_range": effective_config.get("video_conversion_range"),
            "meta_real_width":        effective_config.get("meta_real_width"),
            "track_max_individuals":  effective_config.get("track_max_individuals"),
            "individual_prefix":      effective_config.get("individual_prefix"),
            "existing_clips":         _list_existing_clips(state, video_name),
        })

    @app.post("/tuning/conversion-range")
    def update_conversion_range(request: UpdateConversionRangeRequest):
        """
        Set or clear video_conversion_range for this tuning video only.
        Always written as a video_overrides entry — never at project or
        batch level, since tuning clips have their own framing/timing.
        """
        project_config = _read_project_config(state)
        video_name     = state["video_name"]

        if "video_overrides" not in project_config:
            project_config["video_overrides"] = {}
        if video_name not in project_config["video_overrides"]:
            project_config["video_overrides"][video_name] = {}

        if request.video_conversion_range is None:
            project_config["video_overrides"][video_name].pop("video_conversion_range", None)
        else:
            project_config["video_overrides"][video_name]["video_conversion_range"] = \
                request.video_conversion_range

        _write_project_config(state, project_config)
        return JSONResponse({"ok": True})

    @app.post("/tuning/save-to-project")
    def save_to_project(request: SaveValuesRequest):
        """
        Write confirmed_detect_threshold / animal_size_min / animal_size_max
        as project-level defaults in project.yaml.
        """
        project_config = _read_project_config(state)
        if request.confirmed_detect_threshold is not None:
            project_config["confirmed_detect_threshold"] = request.confirmed_detect_threshold
        if request.animal_size_min is not None:
            project_config["animal_size_min"] = request.animal_size_min
        if request.animal_size_max is not None:
            project_config["animal_size_max"] = request.animal_size_max
        _write_project_config(state, project_config)
        logger.info(
            "Saved to project defaults: threshold=%s, size=[%s,%s]",
            request.confirmed_detect_threshold, request.animal_size_min, request.animal_size_max,
        )
        return JSONResponse({"ok": True})

    @app.post("/tuning/save-to-batch")
    def save_to_batch(request: SaveToBatchRequest):
        """
        Write confirmed_detect_threshold / animal_size_min / animal_size_max
        into a specific batch's config in project.yaml. The batch is chosen
        freely from the dropdown — it does not need to contain the current
        tuning video.
        """
        project_config = _read_project_config(state)
        batches_config = project_config.get("batches") or {}
        if request.batch_name not in batches_config:
            return JSONResponse(
                {"error": f"Batch '{request.batch_name}' not found"}, status_code=404
            )

        batch_config = batches_config[request.batch_name]
        if request.confirmed_detect_threshold is not None:
            batch_config["confirmed_detect_threshold"] = request.confirmed_detect_threshold
        if request.animal_size_min is not None:
            batch_config["animal_size_min"] = request.animal_size_min
        if request.animal_size_max is not None:
            batch_config["animal_size_max"] = request.animal_size_max

        _write_project_config(state, project_config)
        logger.info(
            "Saved to batch '%s': threshold=%s, size=[%s,%s]",
            request.batch_name, request.confirmed_detect_threshold,
            request.animal_size_min, request.animal_size_max,
        )
        return JSONResponse({"ok": True})

    @app.get("/tuning/batches")
    def get_batches_for_tuning():
        """
        List all batch names (for the save-to-batch dropdown) and, if the
        current tuning video happens to belong to one, its name — purely
        informational, the dropdown is not restricted to it.
        """
        project_config = _read_project_config(state)
        batches_config = project_config.get("batches") or {}
        video_name     = state.get("video_name")

        current_video_batch = (
            resolve_batch_for_video(video_name, project_config) if video_name else None
        )

        return JSONResponse({
            "batches":             sorted(batches_config.keys()),
            "current_video_batch": current_video_batch,
        })

    @app.post("/tuning/run-sweep")
    def run_sweep_route(request: RunSweepRequest):
        """
        Run the threshold sweep (blocking). Reuses run_sweep from _sweep.py
        unchanged. Returns the filenames of produced clips. If stopped early
        via /tuning/stop-sweep, returns whatever clips completed before the
        stop request, with cancelled=True.
        """
        from _sweep import run_sweep

        project_dir    = state["project_dir"]
        video_name     = state["video_name"]
        if video_name is None:
            return JSONResponse({"error": "No video selected."}, status_code=400)

        project_config    = _read_project_config(state)
        effective_config  = resolve_effective_config(video_name, project_config)
        effective_config["sweep_thresholds"] = request.thresholds
        if request.animal_size_min is not None:
            effective_config["animal_size_min"] = request.animal_size_min
        if request.animal_size_max is not None:
            effective_config["animal_size_max"] = request.animal_size_max

        tuning_dir = project_dir / "tuning"
        videos_dir = state["videos_dir"]   # already mode-resolved by app.py
        masks_dir  = state["masks_dir"]    # already mode-resolved by app.py

        try:
            clip_files = run_sweep(
                video_name=video_name,
                project_dir=project_dir,
                videos_dir=videos_dir,
                masks_dir=masks_dir,
                tuning_dir=tuning_dir,
                base_settings_file=state["base_settings_file"],
                pipeline_config=state.get("pipeline_config", {}),
                effective_config=effective_config,
                project_config=project_config,
            )
        except Exception as error:
            logger.exception("Sweep failed")
            return JSONResponse({"error": str(error)}, status_code=500)

        cancelled = len(clip_files) < len(request.thresholds)
        return JSONResponse({
            "clips":     [clip_file.name for clip_file in clip_files],
            "cancelled": cancelled,
        })

    @app.post("/tuning/stop-sweep")
    def stop_sweep_route():
        """Request cancellation of the currently running sweep, if any."""
        from _sweep import request_sweep_stop
        request_sweep_stop()
        return JSONResponse({"ok": True})

    @app.get("/tuning/clips")
    def list_clips():
        """List all {video}_t*.mp4 clips currently in the tuning folder."""
        return JSONResponse({
            "clips": _list_existing_clips(state, state.get("video_name")),
        })

    @app.get("/tuning/clip/{filename}")
    def get_clip(filename: str):
        """Serve a tuning clip file with range-request support for scrubbing."""
        tuning_dir = state["project_dir"] / "tuning"
        clip_file  = tuning_dir / filename
        if not clip_file.exists() or clip_file.parent != tuning_dir:
            return JSONResponse({"error": "Clip not found"}, status_code=404)
        return FileResponse(path=clip_file, media_type="video/mp4")

    @app.get("/tuning/clip-settings/{filename}")
    def get_clip_settings(filename: str):
        """
        Read the .settings file that produced a given clip and return the
        values relevant to display in the sidebar — currently
        detect_size_filter (animal size min/max). Extensible to other
        settings later without changing the clip filename scheme.

        Clip filenames are {video_name}_t{threshold:03d}.mp4; the matching
        settings file is {video_name}_thresh{threshold:03d}.settings.
        """
        import re
        match = re.match(r"^(.+)_t(\d+)\.mp4$", filename)
        if not match:
            return JSONResponse({"error": "Could not parse filename"}, status_code=400)

        video_name, threshold_str = match.group(1), match.group(2)
        tuning_dir     = state["project_dir"] / "tuning"
        settings_file  = tuning_dir / f"{video_name}_thresh{threshold_str}.settings"

        if not settings_file.exists():
            return JSONResponse({
                "found": False,
                "animal_size_min": None,
                "animal_size_max": None,
            })

        from _settings import read_settings
        settings = read_settings(settings_file)
        raw = settings.get("detect_size_filter")
        size_min, size_max = None, None
        if raw:
            try:
                import json
                parsed = json.loads(raw)
                if parsed and isinstance(parsed[0], list) and len(parsed[0]) == 2:
                    size_min, size_max = parsed[0]
            except (ValueError, TypeError, IndexError):
                pass

        return JSONResponse({
            "found": True,
            "animal_size_min": size_min,
            "animal_size_max": size_max,
        })


# =============================================================================
# Helpers
# =============================================================================

def _read_project_config(state: dict) -> dict:
    return yaml.safe_load(state["project_yaml_file"].read_text())


def _write_project_config(state: dict, config: dict) -> None:
    state["project_yaml_file"].write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )


def _list_existing_clips(state: dict, video_name: str | None) -> list[str]:
    """
    List all *_t*.mp4 clips in the tuning folder, most relevant ones (for
    the current video) first.
    """
    tuning_dir = state["project_dir"] / "tuning"
    if not tuning_dir.exists():
        return []
    all_clips = sorted(f.name for f in tuning_dir.glob("*_t*.mp4"))
    if video_name:
        current_video_clips = [c for c in all_clips if c.startswith(f"{video_name}_t")]
        other_clips         = [c for c in all_clips if not c.startswith(f"{video_name}_t")]
        return current_video_clips + other_clips
    return all_clips


# =============================================================================
# HTML fragment
# =============================================================================

def build_tune_tab_html(video_name: str) -> str:
    return f"""
<div style="display:flex; width:100%; height:100%;">

  <div id="tune-sidebar" style="width:280px; min-width:280px; background:#16213e;
       display:flex; flex-direction:column; padding:12px; gap:10px;
       overflow-y:auto; border-right:1px solid #0f3460;">

    <h2 style="font-size:13px; color:#a0c4ff; text-transform:uppercase; letter-spacing:1px;">
      Tuning</h2>
    <div id="tune-status" style="font-size:11px; color:#7ec8e3; padding:6px;
         background:#0f3460; border-radius:4px; min-height:32px;">Loading…</div>

    <div style="background:#0f3460; border-radius:6px; padding:10px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">Sweep parameters</h3>

      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">
          Threshold values (comma-separated)</label>
        <input type="text" id="tune-thresholds" placeholder="e.g. 40,50,60"
               style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
               border-radius:4px; color:#e0e0e0; font-size:12px;">
      </div>

      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">
          Animal size — min / max (px)</label>
        <div style="display:flex; gap:6px;">
          <input type="number" id="tune-size-min" placeholder="min"
                 style="width:50%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;">
          <input type="number" id="tune-size-max" placeholder="max"
                 style="width:50%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;">
        </div>
      </div>

      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">
          Conversion range [start,end] (this clip only)</label>
        <input type="text" id="tune-conversion-range" placeholder="e.g. 0,900"
               style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
               border-radius:4px; color:#e0e0e0; font-size:12px;"
               onblur="tuneSaveConversionRange()">
      </div>

      <button onclick="tuneRunSweep()" id="tune-run-btn" style="width:100%; padding:8px 10px;
              border:none; border-radius:5px; cursor:pointer; font-size:12px; font-weight:700;
              background:#e76f51; color:#fff; margin-top:4px;">▶ Run sweep</button>
      <button onclick="tuneStopSweep()" id="tune-stop-btn" style="display:none; width:100%;
              padding:8px 10px; border:none; border-radius:5px; cursor:pointer; font-size:12px;
              font-weight:700; background:#9d0208; color:#ffccd5; margin-top:6px;">⏹ Stop</button>
    </div>

    <div style="background:#0f3460; border-radius:6px; padding:10px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">
        Existing clips (double-click to load)</h3>
      <div id="tune-clip-list" style="max-height:220px; overflow-y:auto; font-size:11px;"></div>
    </div>

    <div style="background:#0f3460; border-radius:6px; padding:10px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">Selected clip's values</h3>
      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">Threshold</label>
        <input type="number" id="tune-selected-threshold" placeholder="select a clip"
               style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
               border-radius:4px; color:#e0e0e0; font-size:12px;">
      </div>
      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">Animal size — min / max (px)</label>
        <div style="display:flex; gap:6px;">
          <input type="number" id="tune-selected-size-min" placeholder="min"
                 style="width:50%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;">
          <input type="number" id="tune-selected-size-max" placeholder="max"
                 style="width:50%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;">
        </div>
      </div>
    </div>

    <div style="background:#0f3460; border-radius:6px; padding:10px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">Save to project</h3>
      <button onclick="tuneSaveToProject()" style="width:100%; padding:7px 10px; border:none;
              border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
              background:#2d6a4f; color:#d8f3dc;">💾 Save to project defaults</button>
    </div>

    <div style="background:#0f3460; border-radius:6px; padding:10px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">Save to batch</h3>
      <div id="tune-clip-batch-note" style="font-size:10px; color:#789; margin-bottom:6px;"></div>
      <select id="tune-batch-select" style="width:100%; padding:5px 6px; background:#1a1a2e;
              border:1px solid #415a77; border-radius:4px; color:#e0e0e0; font-size:12px;
              margin-bottom:6px;">
        <option value="">Loading batches…</option>
      </select>
      <button onclick="tuneSaveToBatch()" style="width:100%; padding:7px 10px; border:none;
              border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
              background:#6a0572; color:#f3e5f5;">💾 Save to selected batch</button>
    </div>
  </div>

  <div style="flex:1; display:flex; flex-direction:column; padding:16px; overflow:hidden;">
    <div id="tune-grid" style="flex:1; display:grid; grid-template-columns:1fr 1fr;
         grid-template-rows:1fr 1fr; gap:10px; min-height:0;"></div>

    <div style="display:flex; align-items:center; gap:10px; padding:12px 4px 4px;">
      <button id="tune-play-btn" onclick="tuneTogglePlay()" style="width:auto; padding:8px 16px;
              border:none; border-radius:5px; cursor:pointer; font-size:14px; font-weight:700;
              background:#2d6a4f; color:#d8f3dc;">▶</button>
      <input type="range" id="tune-scrub" min="0" max="100" value="0" step="0.1"
             style="flex:1;" oninput="tuneScrub(this.value)">
      <span id="tune-time-label" style="font-size:11px; color:#aaa; width:90px; text-align:right;">0:00 / 0:00</span>
    </div>
  </div>
</div>

<script>
(function() {{
  const VIDEO_NAME = {('"' + video_name + '"') if video_name else 'null'};
  let tuneSlots = [null, null, null, null];   // each: {{filename}} or null
  let tuneSelectedSlot = null;
  let tunePlaying = false;
  let tuneSyncing = false;

  window.tuneInit = async function() {{
    await tuneLoadConfig();
    await loadBatchSelector();
    renderGrid();
    setupMasterScrubLoop();
  }};

  async function tuneLoadConfig() {{
    const res = await fetch("/tuning/config"), d = await res.json();
    document.getElementById("tune-thresholds").value = (d.sweep_thresholds || []).join(",");
    if (d.animal_size_min != null) document.getElementById("tune-size-min").value = d.animal_size_min;
    if (d.animal_size_max != null) document.getElementById("tune-size-max").value = d.animal_size_max;
    if (d.video_conversion_range) document.getElementById("tune-conversion-range").value = d.video_conversion_range.join(",");
    renderClipList(d.existing_clips || []);
    tuneSetStatus(d.video_name ? `Ready — ${{d.video_name}}` : "No video selected.");
  }}

  window.tuneSaveConversionRange = async function() {{
    const raw = document.getElementById("tune-conversion-range").value.trim();
    let range = null;
    if (raw) {{
      const parts = raw.split(",").map(s => parseInt(s.trim()));
      if (parts.length === 2 && !parts.some(isNaN)) range = parts;
    }}
    await fetch("/tuning/conversion-range", {{method:"POST", headers:{{"Content-Type":"application/json"}},
      body: JSON.stringify({{video_conversion_range: range}})}});
    tuneSetStatus("Conversion range saved for this clip.");
  }};

  window.tuneRunSweep = async function() {{
    const thresholdsRaw = document.getElementById("tune-thresholds").value.trim();
    const thresholds = thresholdsRaw.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    if (thresholds.length === 0) {{ tuneSetStatus("Enter at least one threshold value."); return; }}

    const sizeMin = document.getElementById("tune-size-min").value;
    const sizeMax = document.getElementById("tune-size-max").value;

    const btn     = document.getElementById("tune-run-btn");
    const stopBtn = document.getElementById("tune-stop-btn");
    btn.disabled = true; btn.textContent = "Running… (this can take a while)";
    stopBtn.style.display = "";
    tuneSetStatus(`Running sweep for thresholds: ${{thresholds.join(", ")}}…`);

    try {{
      const res = await fetch("/tuning/run-sweep", {{method:"POST", headers:{{"Content-Type":"application/json"}},
        body: JSON.stringify({{
          thresholds,
          animal_size_min: sizeMin === "" ? null : parseInt(sizeMin),
          animal_size_max: sizeMax === "" ? null : parseInt(sizeMax),
        }})}});
      const d = await res.json();
      if (d.error) {{ tuneSetStatus("Sweep failed: " + d.error); return; }}

      // Auto-populate empty slots with the new clips (up to 4 total)
      let slotIndex = tuneSlots.findIndex(s => s === null);
      for (const filename of d.clips) {{
        if (slotIndex === -1) break;
        tuneSlots[slotIndex] = {{filename}};
        slotIndex = tuneSlots.findIndex(s => s === null);
      }}
      renderGrid();
      tuneSetStatus(d.cancelled
        ? `Sweep stopped — ${{d.clips.length}} clip(s) completed before stopping.`
        : `Sweep complete: ${{d.clips.length}} clip(s) produced.`);
      const clipsRes = await fetch("/tuning/clips"), clipsData = await clipsRes.json();
      renderClipList(clipsData.clips || []);
    }} finally {{
      btn.disabled = false; btn.textContent = "▶ Run sweep";
      stopBtn.style.display = "none";
    }}
  }};

  window.tuneStopSweep = async function() {{
    tuneSetStatus("Stopping sweep…");
    await fetch("/tuning/stop-sweep", {{method:"POST"}});
  }};

  function renderClipList(clips) {{
    const list = document.getElementById("tune-clip-list");
    list.innerHTML = "";
    if (clips.length === 0) {{
      list.innerHTML = '<div style="color:#5a7a99; padding:4px;">No clips yet.</div>';
      return;
    }}
    clips.forEach(filename => {{
      const row = document.createElement("div");
      row.textContent = filename;
      row.style.padding = "4px 6px"; row.style.borderRadius = "4px";
      row.style.cursor = "pointer"; row.style.color = "#ccc";
      row.onmouseenter = () => row.style.background = "#12233f";
      row.onmouseleave = () => row.style.background = "transparent";
      row.ondblclick = () => loadClipIntoSlot(filename);
      list.appendChild(row);
    }});
  }}

  function loadClipIntoSlot(filename) {{
    let targetSlot = tuneSelectedSlot;
    if (targetSlot === null) {{
      targetSlot = tuneSlots.findIndex(s => s === null);
      if (targetSlot === -1) targetSlot = 0;   // grid full: overwrite slot 0
    }}
    tuneSlots[targetSlot] = {{filename}};
    tuneSelectedSlot = targetSlot;
    renderGrid();
    populateSelectedValues(filename);
    tuneSetStatus(`Loaded ${{filename}} into slot ${{targetSlot + 1}}.`);
  }}

  function parseThresholdFromFilename(filename) {{
    // Clips are named {{video_name}}_t{{threshold:03d}}.mp4
    const match = filename.match(/_t(\\d+)\\.mp4$/);
    return match ? parseInt(match[1], 10) : null;
  }}

  async function populateSelectedValues(filename) {{
    const threshold = parseThresholdFromFilename(filename);
    document.getElementById("tune-selected-threshold").value = threshold ?? "";

    // Read the actual size used for this specific clip from its .settings
    // file, rather than whatever currently sits in the sweep-parameters
    // sidebar (which may have changed since this clip was produced).
    document.getElementById("tune-selected-size-min").value = "";
    document.getElementById("tune-selected-size-max").value = "";
    try {{
      const res = await fetch(`/tuning/clip-settings/${{filename}}`);
      const d = await res.json();
      if (d.found) {{
        document.getElementById("tune-selected-size-min").value = d.animal_size_min ?? "";
        document.getElementById("tune-selected-size-max").value = d.animal_size_max ?? "";
      }}
    }} catch (err) {{
      // Settings file missing/unreadable — leave size fields blank rather
      // than silently showing a stale or misleading value.
    }}
  }}

  function readSelectedValues() {{
    const thresholdRaw = document.getElementById("tune-selected-threshold").value;
    const sizeMinRaw   = document.getElementById("tune-selected-size-min").value;
    const sizeMaxRaw   = document.getElementById("tune-selected-size-max").value;
    return {{
      confirmed_detect_threshold: thresholdRaw === "" ? null : parseInt(thresholdRaw),
      animal_size_min: sizeMinRaw === "" ? null : parseInt(sizeMinRaw),
      animal_size_max: sizeMaxRaw === "" ? null : parseInt(sizeMaxRaw),
    }};
  }}

  window.tuneSaveToProject = async function() {{
    const values = readSelectedValues();
    if (values.confirmed_detect_threshold === null) {{
      tuneSetStatus("Select a clip first (its threshold fills the field above).");
      return;
    }}
    await fetch("/tuning/save-to-project", {{method:"POST", headers:{{"Content-Type":"application/json"}},
      body: JSON.stringify(values)}});
    tuneSetStatus(`Saved threshold=${{values.confirmed_detect_threshold}}, ` +
      `size=[${{values.animal_size_min}},${{values.animal_size_max}}] to project defaults.`);
  }};

  window.tuneSaveToBatch = async function() {{
    const batchName = document.getElementById("tune-batch-select").value;
    if (!batchName) {{ tuneSetStatus("Select a batch first."); return; }}
    const values = readSelectedValues();
    if (values.confirmed_detect_threshold === null) {{
      tuneSetStatus("Select a clip first (its threshold fills the field above).");
      return;
    }}
    await fetch("/tuning/save-to-batch", {{method:"POST", headers:{{"Content-Type":"application/json"}},
      body: JSON.stringify({{...values, batch_name: batchName}})}});
    tuneSetStatus(`Saved threshold=${{values.confirmed_detect_threshold}}, ` +
      `size=[${{values.animal_size_min}},${{values.animal_size_max}}] to batch '${{batchName}}'.`);
  }};

  async function loadBatchSelector() {{
    const res = await fetch("/tuning/batches"), d = await res.json();
    const select = document.getElementById("tune-batch-select");
    select.innerHTML = "";
    if (d.batches.length === 0) {{
      select.innerHTML = '<option value="">No batches yet</option>';
    }} else {{
      d.batches.forEach(name => {{
        const opt = document.createElement("option");
        opt.value = name; opt.textContent = name;
        select.appendChild(opt);
      }});
    }}
    const note = document.getElementById("tune-clip-batch-note");
    note.textContent = d.current_video_batch
      ? `This clip's source video is part of batch '${{d.current_video_batch}}'.`
      : "This clip is not part of any batch — you can still save to any batch above.";
  }}

  function renderGrid() {{
    const grid = document.getElementById("tune-grid");
    grid.innerHTML = "";
    tuneSlots.forEach((slot, index) => {{
      const cell = document.createElement("div");
      cell.style.position = "relative"; cell.style.background = "#000";
      cell.style.borderRadius = "6px"; cell.style.overflow = "hidden";
      cell.style.border = (index === tuneSelectedSlot) ? "3px solid #a0c4ff" : "2px solid #333";
      cell.style.display = "flex"; cell.style.alignItems = "center"; cell.style.justifyContent = "center";
      cell.style.cursor = "pointer";
      cell.onclick = () => {{
        tuneSelectedSlot = index;
        renderGrid();
        if (slot) populateSelectedValues(slot.filename);
      }};

      if (slot) {{
        const video = document.createElement("video");
        video.id = `tune-video-${{index}}`;
        video.src = `/tuning/clip/${{slot.filename}}`;
        video.preload = "metadata";
        video.style.width = "100%"; video.style.height = "100%"; video.style.objectFit = "contain";
        video.muted = true;
        cell.appendChild(video);
        enableCellZoom(cell, video);

        const label = document.createElement("span");
        label.textContent = slot.filename;
        label.style.position = "absolute"; label.style.top = "4px"; label.style.left = "4px";
        label.style.fontSize = "10px"; label.style.background = "rgba(0,0,0,0.7)";
        label.style.color = "#7ec8e3"; label.style.padding = "2px 6px"; label.style.borderRadius = "3px";
        cell.appendChild(label);
      }} else {{
        const placeholder = document.createElement("span");
        placeholder.textContent = "Empty slot — double-click a clip on the left to load";
        placeholder.style.fontSize = "11px"; placeholder.style.color = "#445";
        placeholder.style.padding = "0 16px"; placeholder.style.textAlign = "center";
        cell.appendChild(placeholder);
      }}
      grid.appendChild(cell);
    }});
  }}

  function getLoadedVideos() {{
    return tuneSlots
      .map((slot, index) => slot ? document.getElementById(`tune-video-${{index}}`) : null)
      .filter(v => v !== null);
  }}

  window.tuneTogglePlay = function() {{
    const videos = getLoadedVideos();
    if (videos.length === 0) return;
    tunePlaying = !tunePlaying;
    document.getElementById("tune-play-btn").textContent = tunePlaying ? "⏸" : "▶";
    videos.forEach(v => tunePlaying ? v.play() : v.pause());
  }};

  window.tuneScrub = function(value) {{
    const videos = getLoadedVideos();
    if (videos.length === 0) return;
    const maxDuration = Math.max(...videos.map(v => v.duration || 0));
    const targetTime = (value / 100) * maxDuration;
    tuneSyncing = true;
    videos.forEach(v => {{ v.currentTime = targetTime; }});
    tuneSyncing = false;
    updateTimeLabel(targetTime, maxDuration);
  }};

  function setupMasterScrubLoop() {{
    // Drive the scrub bar from the first loaded video's playback position.
    setInterval(() => {{
      if (tuneSyncing) return;
      const videos = getLoadedVideos();
      if (videos.length === 0) return;
      const reference = videos[0];
      const maxDuration = Math.max(...videos.map(v => v.duration || 0)) || 1;
      const pct = (reference.currentTime / maxDuration) * 100;
      const scrub = document.getElementById("tune-scrub");
      if (!scrub.matches(":active")) scrub.value = pct;
      updateTimeLabel(reference.currentTime, maxDuration);

      // Keep videos roughly in sync (resync if drift exceeds 150ms)
      videos.forEach(v => {{
        if (Math.abs(v.currentTime - reference.currentTime) > 0.15) {{
          v.currentTime = reference.currentTime;
        }}
      }});
    }}, 200);
  }}

  function updateTimeLabel(current, total) {{
    const fmt = (s) => {{
      const m = Math.floor(s / 60), sec = Math.floor(s % 60);
      return `${{m}}:${{sec.toString().padStart(2,"0")}}`;
    }};
    document.getElementById("tune-time-label").textContent = `${{fmt(current)}} / ${{fmt(total)}}`;
  }}

  // ============================================================
  // Ctrl+wheel zoom — same pattern as the background tab, scoped locally
  // since this tab's script runs in its own closure.
  // ============================================================
  function enableCellZoom(cell, media) {{
    let zoom = 1;
    const ZOOM_MIN = 1, ZOOM_MAX = 6, ZOOM_STEP = 0.1;

    cell.addEventListener("wheel", (e) => {{
      if (!e.ctrlKey) return;
      e.preventDefault();
      const delta = e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
      const newZoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom + delta));
      if (newZoom === zoom) return;
      zoom = newZoom;
      media.style.transform = `scale(${{zoom}})`;
      media.style.transformOrigin = "center center";
    }}, {{passive: false}});

    cell.addEventListener("dblclick", (e) => {{
      e.stopPropagation();
      zoom = 1;
      media.style.transform = "none";
    }});
  }}

  function tuneSetStatus(msg) {{ document.getElementById("tune-status").textContent = msg; }}

  document.addEventListener("keydown", e => {{
    if (e.target.tagName === "INPUT") return;
    if (activeTab !== "tune") return;
    if ((e.key === "Delete" || e.key === "Backspace") && tuneSelectedSlot !== null) {{
      tuneSlots[tuneSelectedSlot] = null;
      renderGrid();
    }}
  }});

  window.tuneInit();
}})();
</script>
"""
