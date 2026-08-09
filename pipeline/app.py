"""
app.py — Unified pipeline app: a single browser-based tool with one tab per
pipeline step (background, masks, tune, track). Loads the project once;
switching tabs is instant since everything stays in the same page and the
same backend process. A video picker in the topbar lets you switch which
video background/masks operate on without restarting the app.

Usage:
    uv run python pipeline/app.py --project /path/to/project --video pain_test
    uv run python pipeline/app.py --project /path/to/project --video pain_test --tab masks
    uv run python pipeline/app.py --project /path/to/project

If --video is omitted, the app starts with the first video found in
1_videos/ and you can switch from the in-browser picker at any time.

--tab selects which tab is visible on load (default: background).
Available tabs: background, masks. (tune, track planned.)

Press Q in the browser or Ctrl+C in the terminal to quit.
"""

import argparse
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ._tab_background import build_background_tab_html, register_background_routes
from ._tab_masks import build_masks_tab_html, register_masks_routes
from ._tab_batches import build_batches_tab_html, register_batches_routes
from ._tab_tune import build_tune_tab_html, register_tune_routes

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# =============================================================================
# Shared app state — populated once in main(), read by all tab modules
# =============================================================================

app = FastAPI()
APP_STATE: dict = {}
_server: uvicorn.Server | None = None

AVAILABLE_TABS = ["background", "masks", "batches", "tune"]


class SelectVideoRequest(BaseModel):
    video_name: str


class SelectModeRequest(BaseModel):
    mode: str   # "project" | "tuning"


def tracking_root(project_dir: Path) -> Path:
    """
    All tracker data lives under <project>/2_tracking/, leaving the project
    root for project.yaml, the pipeline notebook, and analysis folders.
    """
    return project_dir / "2_tracking"


def _base_dirs(state: dict) -> tuple[Path, Path, Path]:
    """
    Return (videos_dir, pv_dir, masks_dir) for the current mode.

    project mode: 2_tracking/1_videos/, 2_tracking/2_pv/, 2_tracking/masks/
    tuning mode:  2_tracking/tuning/{1_videos,2_pv,masks}/
    """
    root = tracking_root(state["project_dir"])
    if state.get("mode") == "tuning":
        tuning_dir = root / "tuning"
        return (
            tuning_dir / "1_videos",
            tuning_dir / "2_pv",
            tuning_dir / "masks",
        )
    return (
        root / "1_videos",
        root / "2_pv",
        root / "masks",
    )


# =============================================================================
# Shell-level routes (shutdown, SSE, index page)
# =============================================================================

@app.post("/shutdown")
def shutdown():
    def _stop():
        time.sleep(0.2)
        if _server:
            _server.should_exit = True
    threading.Thread(target=_stop, daemon=True).start()
    return JSONResponse({"ok": True})


@app.get("/events")
def sse_events():
    def stream():
        try:
            while True:
                yield "data: ping\n\n"
                time.sleep(2)
        except GeneratorExit:
            pass
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    if APP_STATE.get("needs_setup"):
        return HTMLResponse(_build_setup_html())
    return HTMLResponse(_build_shell_html())


@app.get("/videos")
def list_videos():
    """List videos available in the current mode's videos folder."""
    videos_dir, _, _ = _base_dirs(APP_STATE)
    video_extension   = APP_STATE["video_extension"]
    videos = _list_videos(videos_dir, video_extension)
    return JSONResponse({
        "videos":  videos,
        "current": APP_STATE.get("video_name"),
        "mode":    APP_STATE.get("mode", "project"),
    })


@app.post("/link-video")
def link_video(payload: dict):
    """
    Create a symbolic link in the current mode's 1_videos/ folder pointing
    to a video file on an external drive or elsewhere on disk.
    The source file must exist at the time the link is created.
    """
    source_path = payload.get("source_path", "").strip()
    if not source_path:
        return JSONResponse({"error": "source_path is required."}, status_code=400)

    source = Path(source_path).resolve()
    if not source.exists():
        return JSONResponse(
            {"error": f"File not found: {source}. Is the drive mounted?"}, status_code=400
        )
    if not source.is_file():
        return JSONResponse(
            {"error": f"{source} is not a file."}, status_code=400
        )

    videos_dir, _, _ = _base_dirs(APP_STATE)
    videos_dir.mkdir(parents=True, exist_ok=True)

    link_target = videos_dir / source.name
    if link_target.exists() or link_target.is_symlink():
        return JSONResponse(
            {"error": f"A file or link named '{source.name}' already exists in {videos_dir}. "
                      "Rename the source file or remove the existing link first."},
            status_code=400,
        )

    link_target.symlink_to(source)
    logger.info("Created symlink: %s → %s", link_target, source)

    # Re-detect video extension now that a file exists
    project_config = yaml.safe_load(APP_STATE["project_yaml_file"].read_text())
    APP_STATE["video_extension"] = _detect_video_extension(
        videos_dir, override=project_config.get("video_extension")
    )

    return JSONResponse({"ok": True, "link": str(link_target), "target": str(source)})


@app.post("/select-video")
def select_video(request: SelectVideoRequest):
    """
    Switch the shared selected video (used by background + masks tabs).
    Reloads frame/fps/total_frames into APP_STATE.
    """
    try:
        _load_video_into_state(request.video_name)
    except (FileNotFoundError, RuntimeError) as error:
        return JSONResponse({"error": str(error)}, status_code=404)
    return JSONResponse({"ok": True, "video_name": request.video_name})


@app.post("/select-mode")
def select_mode(request: SelectModeRequest):
    """
    Switch between 'project' and 'tuning' mode. This changes which folders
    videos/background/masks are read from and written to. The currently
    selected video is cleared; the frontend should reload to pick a new one.
    """
    if request.mode not in ("project", "tuning"):
        return JSONResponse({"error": "mode must be 'project' or 'tuning'"}, status_code=400)

    APP_STATE["mode"] = request.mode
    videos_dir, pv_dir, masks_dir = _base_dirs(APP_STATE)
    videos_dir.mkdir(parents=True, exist_ok=True)
    pv_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    APP_STATE["videos_dir"]      = videos_dir
    APP_STATE["pv_dir"]          = pv_dir
    APP_STATE["masks_dir"]       = masks_dir
    APP_STATE["video_extension"] = _detect_video_extension(
        videos_dir,
        override=yaml.safe_load(
            APP_STATE["project_yaml_file"].read_text()
        ).get("video_extension"),
    )

    available_videos = _list_videos(videos_dir, APP_STATE["video_extension"])
    if available_videos:
        _load_video_into_state(available_videos[0])
    else:
        APP_STATE["video_name"] = None

    return JSONResponse({
        "ok": True,
        "mode": request.mode,
        "video_name": APP_STATE.get("video_name"),
        "has_videos": bool(available_videos),
    })


# =============================================================================
# Shell HTML — page chrome, tab switcher, shared CSS/JS; each tab module
# contributes its own HTML fragment + JS namespace.
# =============================================================================

def _build_setup_html() -> str:
    project_dir = APP_STATE["project_dir"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>New project — {project_dir.name}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0;
       display: flex; align-items: center; justify-content: center;
       min-height: 100vh; }}
.card {{ background: #16213e; border: 1px solid #0f3460; border-radius: 12px;
         padding: 32px; width: 460px; }}
h1 {{ font-size: 16px; color: #a0c4ff; text-transform: uppercase;
      letter-spacing: 1px; margin-bottom: 6px; }}
.notice {{ font-size: 12px; color: #7ec8e3; background: #0f3460;
           border-radius: 6px; padding: 10px 14px; margin-bottom: 24px;
           line-height: 1.6; }}
.notice code {{ color: #ffd700; font-size: 11px; }}
label {{ font-size: 11px; color: #aaa; display: block; margin-bottom: 3px; }}
input {{ width: 100%; padding: 7px 9px; background: #1a1a2e; border: 1px solid #415a77;
         border-radius: 5px; color: #e0e0e0; font-size: 13px; margin-bottom: 14px; }}
.row {{ display: flex; gap: 10px; }}
.row > div {{ flex: 1; }}
button {{ width: 100%; padding: 10px; border: none; border-radius: 6px;
          background: #2d6a4f; color: #d8f3dc; font-size: 13px; font-weight: 700;
          cursor: pointer; margin-top: 6px; }}
button:hover {{ opacity: 0.88; }}
#err {{ font-size: 12px; color: #ff6b6b; margin-top: 10px; min-height: 18px; }}
</style>
</head>
<body>
<div class="card">
  <h1>New project</h1>
  <div class="notice">
    No project found at<br>
    <code>{project_dir}</code><br><br>
    Fill in the required fields below. The folder structure will be created
    for you and the app will launch immediately.
  </div>

  <label>Arena width (cm) *</label>
  <input type="number" id="meta-real-width" step="0.1" placeholder="e.g. 25.4">

  <div class="row">
    <div>
      <label>Max individuals *</label>
      <input type="number" id="max-individuals" min="1" placeholder="e.g. 8">
    </div>
    <div>
      <label>Individual prefix *</label>
      <input type="text" id="individual-prefix" value="ant" placeholder="ant">
    </div>
  </div>

  <label>Video file extension</label>
  <input type="text" id="video-extension" value="MP4" placeholder="MP4">

  <button onclick="createProject()">Create project and launch →</button>
  <div id="err"></div>
</div>
<script>
async function createProject() {{
  const width    = document.getElementById("meta-real-width").value.trim();
  const maxInd   = document.getElementById("max-individuals").value.trim();
  const prefix   = document.getElementById("individual-prefix").value.trim();
  const ext      = document.getElementById("video-extension").value.trim();
  const errEl    = document.getElementById("err");
  errEl.textContent = "";

  if (!width || !maxInd || !prefix) {{
    errEl.textContent = "Please fill in all required fields.";
    return;
  }}

  const res = await fetch("/setup/create-project", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{
      meta_real_width:       parseFloat(width),
      track_max_individuals: parseInt(maxInd),
      individual_prefix:     prefix,
      video_extension:       ext || "MP4",
    }}),
  }});
  const d = await res.json();
  if (d.error) {{ errEl.textContent = d.error; return; }}
  window.location.reload();
}}
document.addEventListener("keydown", e => {{
  if (e.key === "Enter") createProject();
}});
</script>
</body>
</html>"""


def register_setup_routes(app: FastAPI, state: dict) -> None:
    """Routes active only when no project.yaml exists yet."""

    @app.post("/setup/create-project")
    def create_project(payload: dict):
        project_dir       = state["project_dir"]
        project_yaml_file = state["project_yaml_file"]
        pipeline_dir      = Path(__file__).resolve().parent
        template_file     = pipeline_dir.parent / "project_template" / "project.yaml"

        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            for subdir in [
                "2_tracking/1_videos", "2_tracking/2_pv",
                "2_tracking/3_csv_individual", "2_tracking/4_csv_video",
                "2_tracking/masks",
                "2_tracking/tuning/1_videos", "2_tracking/tuning/2_pv",
                "2_tracking/tuning/masks",
                "3_analysis",
            ]:
                (project_dir / subdir).mkdir(parents=True, exist_ok=True)

            template = template_file.read_text()
            template = template.replace("__PROJECT_NAME__", project_dir.name)
            config = yaml.safe_load(template)
            config["meta_real_width"]       = payload.get("meta_real_width")
            config["track_max_individuals"] = payload.get("track_max_individuals")
            config["individual_prefix"]     = payload.get("individual_prefix", "ant")
            config["video_extension"]       = payload.get("video_extension", "MP4")
            project_yaml_file.write_text(
                yaml.dump(config, default_flow_style=False, sort_keys=False)
            )

            # Reload APP_STATE so the normal shell renders correctly
            # when the browser reloads after this response.
            project_config = yaml.safe_load(project_yaml_file.read_text())
            state["needs_setup"]        = False
            state["meta_real_width"]    = project_config.get("meta_real_width")
            state["video_extension"]    = _detect_video_extension(
                tracking_root(project_dir) / "1_videos",
                override=project_config.get("video_extension"),
            )
            state["video_name"]         = None

            # Register the full tab routes now that we have a valid project.
            register_background_routes(app, state)
            register_masks_routes(app, state)
            register_batches_routes(app, state)
            register_tune_routes(app, state)

            logger.info("New project created at %s", project_dir)
            return JSONResponse({"ok": True})

        except Exception as error:
            logger.exception("Failed to create project")
            return JSONResponse({"error": str(error)}, status_code=500)


def _build_no_video_placeholder() -> str:
    """Shown in place of tab content when no video is selected yet."""
    return """
<div style="display:flex; width:100%; height:100%; align-items:center;
     justify-content:center; color:#7ec8e3; font-size:14px; text-align:center;">
  <div>
    No video selected.<br>
    <span style="font-size:12px; color:#5a7a99;">
      Add a video to the current mode's folder, or pick one from the
      dropdown above once one is available.
    </span>
  </div>
</div>
"""


def _build_shell_html() -> str:
    video_name   = APP_STATE["video_name"]
    initial_tab  = APP_STATE["initial_tab"]
    current_mode = APP_STATE.get("mode", "project")

    if video_name is None:
        background_fragment = _build_no_video_placeholder()
        masks_fragment       = _build_no_video_placeholder()
        tune_fragment        = _build_no_video_placeholder()
    else:
        background_fragment = build_background_tab_html(video_name)
        masks_fragment       = build_masks_tab_html(video_name)
        tune_fragment        = build_tune_tab_html(video_name)

    # Batches tab is project-wide and does not depend on a selected video.
    batches_fragment = build_batches_tab_html(video_name)

    page_title = video_name if video_name else "no video selected"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline — {page_title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}

#topbar {{
  display: flex; align-items: center; gap: 4px; padding: 8px 12px;
  background: #0f1530; border-bottom: 1px solid #0f3460;
}}
#topbar h1 {{ font-size: 13px; color: #a0c4ff; text-transform: uppercase;
              letter-spacing: 1px; margin-right: 16px; }}
.tab-btn {{
  padding: 7px 16px; border: none; border-radius: 5px 5px 0 0;
  background: #16213e; color: #8899bb; cursor: pointer; font-size: 12px;
  font-weight: 600; transition: all .15s;
}}
.tab-btn:hover {{ background: #1a2a4e; color: #c0d8ff; }}
.tab-btn.active {{ background: #0f3460; color: #fff; }}
.tab-btn.disabled {{ color: #445; cursor: not-allowed; pointer-events: none; }}
.tab-btn.disabled:hover {{ background: #16213e; }}
#topbar .spacer {{ flex: 1; }}
.btn-close {{ background: #7b2d00; color: #ffe0d0; border: none;
              border-radius: 5px; padding: 7px 14px; cursor: pointer;
              font-size: 12px; font-weight: 600; }}
.btn-close:hover {{ opacity: 0.85; }}

#tab-content {{ flex: 1; overflow: hidden; position: relative; }}
.tab-pane {{ display: none; height: 100%; }}
.tab-pane.active {{ display: flex; }}

#shutdown-overlay {{ display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.75); align-items: center;
  justify-content: center; z-index: 200; }}
#shutdown-overlay.visible {{ display: flex; }}
#shutdown-box {{ background: #16213e; border: 1px solid #a0c4ff; border-radius: 10px;
  padding: 28px 32px; text-align: center; max-width: 340px; }}
#shutdown-box h3 {{ color: #a0c4ff; margin-bottom: 14px; }}
#shutdown-box p  {{ color: #ccc; font-size: 13px; margin-bottom: 20px; }}
.btn-row {{ display: flex; gap: 10px; justify-content: center; }}
.btn-row button {{ width: auto; padding: 8px 20px; border: none; border-radius: 5px;
                    cursor: pointer; font-size: 12px; font-weight: 600; }}
.btn-include {{ background: #2d6a4f; color: #d8f3dc; }}
.btn-ignore  {{ background: #9d0208; color: #ffccd5; }}
.btn-neutral {{ background: #415a77; color: #e0e0e0; }}
</style>
</head>
<body>

<div id="topbar">
  <div id="mode-toggle" style="display:flex; background:#0f1530; border:1px solid #2a4a7a;
       border-radius:5px; overflow:hidden; margin-right:12px;">
    <button id="mode-btn-project" class="mode-btn active" onclick="onModePicked('project')"
            style="padding:6px 12px; border:none; background:#0f3460; color:#fff;
                   font-size:11px; font-weight:600; cursor:pointer;">📁 Project</button>
    <button id="mode-btn-tuning" class="mode-btn" onclick="onModePicked('tuning')"
            style="padding:6px 12px; border:none; background:#16213e; color:#8899bb;
                   font-size:11px; font-weight:600; cursor:pointer;">🧪 Tuning</button>
  </div>
  <select id="video-picker" onchange="onVideoPicked()"
          style="background:#0f3460; color:#fff; border:1px solid #2a4a7a;
                 border-radius:5px; padding:6px 10px; font-size:12px;
                 font-weight:600; margin-right:16px; cursor:pointer;">
    <option value="">Loading videos…</option>
  </select>
  <button class="tab-btn" data-tab="background" onclick="switchTab('background')">🖼 Background</button>
  <button class="tab-btn" data-tab="masks"      onclick="switchTab('masks')">🎯 Masks</button>
  <button class="tab-btn" data-tab="batches"    onclick="switchTab('batches')">⚙️ Parameters</button>
  <button class="tab-btn" data-tab="tune" id="tab-btn-tune"
          onclick="switchTab('tune')">🧪 Tune</button>
  <div class="spacer"></div>
  <button class="btn-neutral" style="font-size:12px; padding:6px 12px; border:none;
          border-radius:5px; cursor:pointer; background:#16213e; color:#8899bb;"
          onclick="showLinkVideoPanel()">🔗 Link video</button>
  <button class="btn-close" onclick="requestClose()">✖ Close (Q)</button>
</div>

<div id="tab-content">
  <div class="tab-pane" id="pane-background">{background_fragment}</div>
  <div class="tab-pane" id="pane-masks">{masks_fragment}</div>
  <div class="tab-pane" id="pane-batches">{batches_fragment}</div>
  <div class="tab-pane" id="pane-tune">{tune_fragment}</div>
</div>

<div id="shutdown-overlay">
  <div id="shutdown-box">
    <h3>Close app?</h3>
    <p id="shutdown-msg">Unsaved changes in some tabs may be lost.</p>
    <div class="btn-row">
      <button class="btn-ignore"  onclick="closeNow()">✖ Close anyway</button>
      <button class="btn-neutral" onclick="cancelClose()">Cancel</button>
    </div>
  </div>
</div>

<div id="link-video-overlay" style="display:none; position:fixed; inset:0;
     background:rgba(0,0,0,0.75); align-items:center; justify-content:center; z-index:200;">
  <div style="background:#16213e; border:1px solid #a0c4ff; border-radius:10px;
       padding:28px 32px; width:480px;">
    <h3 style="color:#a0c4ff; margin-bottom:8px;">🔗 Link external video</h3>
    <p style="font-size:12px; color:#ccc; margin-bottom:16px; line-height:1.6;">
      Creates a symbolic link in the current mode's <code>1_videos/</code> folder
      pointing to a video on an external drive or anywhere on disk.<br>
      The drive must be mounted whenever the app reads that video.
    </p>
    <label style="font-size:11px; color:#aaa; display:block; margin-bottom:3px;">
      Full path to video file</label>
    <input type="text" id="link-video-path" placeholder="/Volumes/MyDrive/video.MP4"
           style="width:100%; padding:7px 9px; background:#1a1a2e; border:1px solid #415a77;
           border-radius:5px; color:#e0e0e0; font-size:12px; margin-bottom:14px;">
    <div id="link-video-err" style="font-size:12px; color:#ff6b6b; margin-bottom:10px;"></div>
    <div style="display:flex; gap:10px; justify-content:flex-end;">
      <button onclick="hideLinkVideoPanel()" style="width:auto; padding:8px 18px; border:none;
              border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
              background:#415a77; color:#e0e0e0;">Cancel</button>
      <button onclick="doLinkVideo()" style="width:auto; padding:8px 18px; border:none;
              border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
              background:#2d6a4f; color:#d8f3dc;">Create link</button>
    </div>
  </div>
</div>

<script>
const VIDEO_NAME   = {('"' + video_name + '"') if video_name else 'null'};
const CURRENT_MODE = "{current_mode}";
let activeTab = "{initial_tab}";

function switchTab(tab) {{
  const btn = document.querySelector(`.tab-btn[data-tab="${{tab}}"]`);
  if (btn && btn.classList.contains("disabled")) {{
    tab = "background";
  }}
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("pane-" + tab).classList.add("active");
  const activeBtn = document.querySelector(`.tab-btn[data-tab="${{tab}}"]`);
  if (activeBtn) activeBtn.classList.add("active");
  activeTab = tab;
  if (window.onTabActivated) window.onTabActivated(tab);
}}

function applyModeTabState() {{
  const tuneBtn = document.getElementById("tab-btn-tune");
  if (CURRENT_MODE === "tuning") {{
    tuneBtn.classList.remove("disabled");
    tuneBtn.title = "";
  }} else {{
    tuneBtn.classList.add("disabled");
    tuneBtn.title = "Switch to Tuning mode to use this tab";
  }}
}}

async function populateVideoPicker() {{
  const res = await fetch("/videos"), d = await res.json();
  const picker = document.getElementById("video-picker");
  picker.innerHTML = "";
  if (d.videos.length === 0) {{
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "No videos found";
    picker.appendChild(opt);
  }}
  d.videos.forEach(name => {{
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === d.current) opt.selected = true;
    picker.appendChild(opt);
  }});
  setModeButtons(d.mode);
}}

function setModeButtons(mode) {{
  const projectBtn = document.getElementById("mode-btn-project");
  const tuningBtn  = document.getElementById("mode-btn-tuning");
  if (mode === "tuning") {{
    tuningBtn.style.background = "#0f3460"; tuningBtn.style.color = "#fff";
    projectBtn.style.background = "#16213e"; projectBtn.style.color = "#8899bb";
  }} else {{
    projectBtn.style.background = "#0f3460"; projectBtn.style.color = "#fff";
    tuningBtn.style.background = "#16213e"; tuningBtn.style.color = "#8899bb";
  }}
}}

async function onModePicked(mode) {{
  setModeButtons(mode);
  const res = await fetch("/select-mode", {{method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{mode}})}});
  const d = await res.json();
  if (!d.ok) {{ alert("Could not switch mode: " + (d.error || "unknown error")); return; }}
  // Reload to re-render both tabs server-side for the new mode (with or
  // without a video — the placeholder pane handles the empty case).
  const url = new URL(window.location.href);
  url.searchParams.set("tab", activeTab);
  window.location.href = url.toString();
}}

async function onVideoPicked() {{
  const picker = document.getElementById("video-picker");
  const selected = picker.value;
  if (selected === VIDEO_NAME) return;   // no change
  picker.disabled = true;
  const res = await fetch("/select-video", {{method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{video_name: selected}})}});
  if (res.ok) {{
    // Reload the page so all tabs re-render server-side for the new video.
    // Preserve the currently active tab across the reload via the URL.
    const url = new URL(window.location.href);
    url.searchParams.set("tab", activeTab);
    window.location.href = url.toString();
  }} else {{
    const d = await res.json();
    alert("Could not switch video: " + (d.error || "unknown error"));
    picker.disabled = false;
  }}
}}

function requestClose() {{
  document.getElementById("shutdown-overlay").classList.add("visible");
}}

function showLinkVideoPanel() {{
  document.getElementById("link-video-path").value = "";
  document.getElementById("link-video-err").textContent = "";
  document.getElementById("link-video-overlay").style.display = "flex";
  setTimeout(() => document.getElementById("link-video-path").focus(), 50);
}}

function hideLinkVideoPanel() {{
  document.getElementById("link-video-overlay").style.display = "none";
}}

async function doLinkVideo() {{
  const path   = document.getElementById("link-video-path").value.trim();
  const errEl  = document.getElementById("link-video-err");
  errEl.textContent = "";
  if (!path) {{ errEl.textContent = "Enter a file path."; return; }}

  const res = await fetch("/link-video", {{method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{source_path: path}})}});
  const d = await res.json();
  if (d.error) {{ errEl.textContent = d.error; return; }}
  hideLinkVideoPanel();
  // Refresh the video picker so the new link appears.
  await populateVideoPicker();
}}
function cancelClose() {{
  document.getElementById("shutdown-overlay").classList.remove("visible");
}}
async function closeNow() {{
  await fetch("/shutdown", {{method:"POST"}});
  document.body.innerHTML =
    '<div style="display:flex;align-items:center;justify-content:center;height:100vh;' +
    'background:#1a1a2e;color:#a0c4ff;font-family:system-ui;font-size:18px">' +
    'App closed — you can close this tab.</div>';
}}

function listenForServerShutdown() {{
  const es = new EventSource("/events");
  es.onerror = () => {{
    es.close();
    document.body.innerHTML =
      '<div style="display:flex;align-items:center;justify-content:center;height:100vh;' +
      'background:#1a1a2e;color:#a0c4ff;font-family:system-ui;font-size:18px">' +
      'Server stopped — you can close this tab.</div>';
  }};
}}

document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT") return;
  if (e.key === "q" || e.key === "Q") requestClose();
  if (e.key === "Enter" && document.getElementById("shutdown-overlay").classList.contains("visible")) {{
    closeNow();
  }}
  if (e.key === "Escape") {{
    hideLinkVideoPanel();
    cancelClose();
  }}
}});

window.onload = () => {{
  applyModeTabState();
  // If the URL carries ?tab=..., it takes precedence (set when switching video)
  const urlTab = new URLSearchParams(window.location.search).get("tab");
  switchTab(urlTab && document.getElementById("pane-" + urlTab) ? urlTab : activeTab);
  populateVideoPicker();
  listenForServerShutdown();
}};
</script>
</body>
</html>"""


# =============================================================================
# Main
# =============================================================================

def _load_video_into_state(video_name: str, frame_index: int | None = None) -> None:
    """
    Load a video's frame/fps/total_frames into APP_STATE.
    Called once at startup (if --video given) and again whenever the user
    picks a different video via /select-video, or switches mode.
    Reads from the current mode's videos folder (project vs tuning).
    """
    videos_dir, _, _ = _base_dirs(APP_STATE)
    video_extension   = APP_STATE["video_extension"]
    video_file = videos_dir / f"{video_name}.{video_extension}"
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    capture      = cv2.VideoCapture(str(video_file))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()

    display_frame_index = frame_index if frame_index is not None else total_frames // 2
    capture = cv2.VideoCapture(str(video_file))
    capture.set(cv2.CAP_PROP_POS_FRAMES, display_frame_index)
    success, display_frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Could not read frame {display_frame_index} from {video_file}")

    APP_STATE["video_name"]    = video_name
    APP_STATE["video_file"]    = video_file
    APP_STATE["fps"]           = fps
    APP_STATE["total_frames"]  = total_frames
    APP_STATE["display_frame"] = display_frame


def _detect_video_extension(videos_dir: Path, override: str | None = None) -> str:
    """
    Determine the video file extension to use for this project.

    Priority:
        1. override — the value from project.yaml's video_extension field, if set
        2. Most common extension found in videos_dir
        3. "MP4" as a last-resort default

    The override is only used when explicitly set in project.yaml (not null).
    Logs what was detected so the user can see what's happening.
    """
    if override and override.lower() not in ("", "null", "none"):
        logger.info("Video extension: '%s' (from project.yaml).", override)
        return override

    if videos_dir.exists():
        extensions: dict[str, int] = {}
        for video_file in videos_dir.iterdir():
            if video_file.is_file() or video_file.is_symlink():
                ext = video_file.suffix.lstrip(".")
                if ext:
                    extensions[ext] = extensions.get(ext, 0) + 1
        if extensions:
            detected = max(extensions, key=extensions.__getitem__)
            logger.info(
                "Video extension: '%s' (auto-detected from %s).", detected, videos_dir
            )
            return detected

    logger.info("Video extension: 'MP4' (default — no videos found yet).")
    return "MP4"


def _list_videos(videos_dir: Path, video_extension: str) -> list[str]:
    """Return video names (without extension) found in videos_dir."""
    if not videos_dir.exists():
        return []
    return sorted(
        video_file.stem for video_file in videos_dir.glob(f"*.{video_extension}")
    )


def main() -> None:
    global _server

    parser = argparse.ArgumentParser(description="Unified pipeline app.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--video",   default=None,
                        help="Video name to open initially. If omitted, "
                             "the first video found in the selected mode's "
                             "folder is used, if any.")
    parser.add_argument("--mode",    choices=["project", "tuning"], default="project",
                        help="Start in 'project' (1_videos/) or 'tuning' "
                             "(tuning/1_videos/) mode (default: project)")
    parser.add_argument("--frame",   type=int, default=None,
                        help="Display frame for masks tab (default: middle frame)")
    parser.add_argument("--tab",     choices=AVAILABLE_TABS, default="background",
                        help="Tab to open on launch")
    parser.add_argument("--port",    type=int, default=8000)
    args = parser.parse_args()

    project_dir = Path(args.project).resolve()
    project_yaml_file = project_dir / "project.yaml"
    needs_setup = not project_dir.exists() or not project_yaml_file.exists()

    if needs_setup:
        logger.info(
            "Project directory or project.yaml not found at %s — "
            "will show new-project setup form in the browser.",
            project_dir,
        )

    tracking_dir = tracking_root(project_dir)
    masks_dir  = tracking_dir / "masks"
    pv_dir     = tracking_dir / "2_pv"
    tuning_dir = tracking_dir / "tuning"

    pipeline_dir         = Path(__file__).resolve().parent
    base_settings_file   = pipeline_dir / "default.settings"
    pipeline_config_file = pipeline_dir / "pipeline.yaml"
    pipeline_config = (
        yaml.safe_load(pipeline_config_file.read_text())
        if pipeline_config_file.exists() else {}
    )
    if not pipeline_config_file.exists():
        logger.warning(
            "pipeline.yaml not found at %s — using defaults (conda_env='trex', "
            "miniforge_dir='~/miniforge3'). Copy pipeline.yaml.example if these "
            "are wrong for this machine.",
            pipeline_config_file,
        )

    # Minimal APP_STATE populated before routes are registered so that the
    # setup route (POST /setup/create-project) can access project_dir.
    APP_STATE["project_dir"]        = project_dir
    APP_STATE["project_yaml_file"]  = project_yaml_file
    APP_STATE["mode"]               = args.mode
    APP_STATE["initial_tab"]        = args.tab
    APP_STATE["base_settings_file"] = base_settings_file
    APP_STATE["pipeline_config"]    = pipeline_config
    APP_STATE["needs_setup"]        = needs_setup

    if needs_setup:
        # Defer all further setup to the browser form. Register only the
        # setup routes now; the normal tab routes are registered after the
        # user completes setup and the page reloads.
        APP_STATE["video_name"]      = None
        APP_STATE["video_extension"] = "MP4"
        APP_STATE["meta_real_width"] = None
        APP_STATE["masks_dir"]       = masks_dir
        APP_STATE["pv_dir"]          = pv_dir
        APP_STATE["videos_dir"]      = tracking_root(project_dir) / "1_videos"
        register_setup_routes(app, APP_STATE)
    else:
        # Normal startup: load project config and scaffold any missing folders.
        project_config  = yaml.safe_load(project_yaml_file.read_text())
        meta_real_width = project_config.get("meta_real_width")
        video_extension = _detect_video_extension(
            tracking_root(project_dir) / "1_videos",
            override=project_config.get("video_extension"),
        )

        masks_dir.mkdir(exist_ok=True)
        pv_dir.mkdir(exist_ok=True)
        (tuning_dir / "1_videos").mkdir(parents=True, exist_ok=True)
        (tuning_dir / "2_pv").mkdir(parents=True, exist_ok=True)
        (tuning_dir / "masks").mkdir(parents=True, exist_ok=True)

        APP_STATE["video_extension"] = video_extension
        APP_STATE["meta_real_width"] = meta_real_width
        APP_STATE["masks_dir"]       = masks_dir
        APP_STATE["pv_dir"]          = pv_dir
        APP_STATE["video_name"]      = None

        videos_dir, mode_pv_dir, mode_masks_dir = _base_dirs(APP_STATE)
        APP_STATE["videos_dir"] = videos_dir
        APP_STATE["pv_dir"]     = mode_pv_dir
        APP_STATE["masks_dir"]  = mode_masks_dir

        if args.video is not None:
            try:
                _load_video_into_state(args.video, args.frame)
            except (FileNotFoundError, RuntimeError) as error:
                logger.error(str(error))
                sys.exit(1)
        else:
            available_videos = _list_videos(videos_dir, video_extension)
            if available_videos:
                first_video = available_videos[0]
                logger.info(
                    "No --video given — starting with '%s'. Use the picker in "
                    "the browser to switch. Available videos: %s",
                    first_video, ", ".join(available_videos),
                )
                _load_video_into_state(first_video, args.frame)
            else:
                logger.warning(
                    "No videos found in %s (mode: %s). The app will start with "
                    "no video selected — add a video to that folder and use "
                    "the picker, or switch mode in the browser.",
                    videos_dir, args.mode,
                )

        register_background_routes(app, APP_STATE)
        register_masks_routes(app, APP_STATE)
        register_batches_routes(app, APP_STATE)
        register_tune_routes(app, APP_STATE)

    def open_browser():
        time.sleep(1.2)
        url = f"http://localhost:{args.port}"
        try:
            subprocess.Popen(["xdg-open", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.info("Open your browser at: %s", url)

    threading.Thread(target=open_browser, daemon=True).start()

    def handle_sigint(signum, frame):
        # Cancel any in-flight sweep immediately (TGrabs/TRex subprocess +
        # the per-threshold loop check), then let uvicorn's own SIGINT
        # handling proceed with its normal graceful shutdown.
        try:
            from ._sweep import request_sweep_stop
            request_sweep_stop()
            logger.info("Ctrl+C: requested cancellation of any running sweep.")
        except ImportError:
            pass
        if _server is not None:
            _server.should_exit = True

    signal.signal(signal.SIGINT, handle_sigint)

    config  = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    _server = uvicorn.Server(config)
    logger.info("Pipeline app at http://localhost:%d  (tab: %s)  (Q in browser or Ctrl+C to quit)",
               args.port, args.tab)
    _server.run()


if __name__ == "__main__":
    main()
