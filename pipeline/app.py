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

sys.path.insert(0, str(Path(__file__).parent))
from _tab_background import build_background_tab_html, register_background_routes
from _tab_masks import build_masks_tab_html, register_masks_routes

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

AVAILABLE_TABS = ["background", "masks"]


class SelectVideoRequest(BaseModel):
    video_name: str


class SelectModeRequest(BaseModel):
    mode: str   # "project" | "tuning"


def _base_dirs(state: dict) -> tuple[Path, Path, Path]:
    """
    Return (videos_dir, pv_dir, masks_dir) for the current mode.

    project mode: 1_videos/, 2_pv/, masks/
    tuning mode:  tuning/1_videos/, tuning/2_pv/, tuning/masks/
    """
    project_dir = state["project_dir"]
    if state.get("mode") == "tuning":
        tuning_dir = project_dir / "tuning"
        return (
            tuning_dir / "1_videos",
            tuning_dir / "2_pv",
            tuning_dir / "masks",
        )
    return (
        project_dir / "1_videos",
        project_dir / "2_pv",
        project_dir / "masks",
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
    APP_STATE["videos_dir"] = videos_dir
    APP_STATE["pv_dir"]     = pv_dir
    APP_STATE["masks_dir"]  = masks_dir

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

    if video_name is None:
        background_fragment = _build_no_video_placeholder()
        masks_fragment       = _build_no_video_placeholder()
    else:
        background_fragment = build_background_tab_html(video_name)
        masks_fragment       = build_masks_tab_html(video_name)

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
  <div class="spacer"></div>
  <button class="btn-close" onclick="requestClose()">✖ Close (Q)</button>
</div>

<div id="tab-content">
  <div class="tab-pane" id="pane-background">{background_fragment}</div>
  <div class="tab-pane" id="pane-masks">{masks_fragment}</div>
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

<script>
const VIDEO_NAME = {('"' + video_name + '"') if video_name else 'null'};
let activeTab = "{initial_tab}";

function switchTab(tab) {{
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("pane-" + tab).classList.add("active");
  document.querySelector(`.tab-btn[data-tab="${{tab}}"]`).classList.add("active");
  activeTab = tab;
  if (window.onTabActivated) window.onTabActivated(tab);
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
}});

window.onload = () => {{
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
    if not project_dir.exists():
        logger.error("Project directory not found: %s", project_dir)
        sys.exit(1)

    project_yaml_file = project_dir / "project.yaml"
    project_config    = yaml.safe_load(project_yaml_file.read_text())
    meta_real_width   = project_config.get("meta_real_width")
    video_extension   = project_config.get("video_extension", "MP4")

    if meta_real_width is None:
        logger.error("meta_real_width not set in project.yaml.")
        sys.exit(1)

    masks_dir = project_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    pv_dir = project_dir / "2_pv"
    pv_dir.mkdir(exist_ok=True)
    tuning_dir = project_dir / "tuning"
    (tuning_dir / "1_videos").mkdir(parents=True, exist_ok=True)
    (tuning_dir / "2_pv").mkdir(parents=True, exist_ok=True)
    (tuning_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Populate shared app state (video-specific fields filled in below)
    APP_STATE["project_dir"]       = project_dir
    APP_STATE["project_yaml_file"] = project_yaml_file
    APP_STATE["video_extension"]   = video_extension
    APP_STATE["meta_real_width"]   = meta_real_width
    APP_STATE["mode"]              = args.mode
    APP_STATE["initial_tab"]       = args.tab
    APP_STATE["video_name"]        = None   # set below if a video is found

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

    # Register routes from each tab module (namespaced internally)
    register_background_routes(app, APP_STATE)
    register_masks_routes(app, APP_STATE)

    def open_browser():
        time.sleep(1.2)
        url = f"http://localhost:{args.port}"
        try:
            subprocess.Popen(["xdg-open", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.info("Open your browser at: %s", url)

    threading.Thread(target=open_browser, daemon=True).start()

    config  = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    _server = uvicorn.Server(config)
    logger.info("Pipeline app at http://localhost:%d  (tab: %s)  (Q in browser or Ctrl+C to quit)",
               args.port, args.tab)
    _server.run()


if __name__ == "__main__":
    main()
