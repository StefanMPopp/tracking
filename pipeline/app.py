"""
app.py — Unified pipeline app: a single browser-based tool with one tab per
pipeline step (background, masks, tune, track). Loads the video and project
config once; switching tabs is instant since everything stays in the same
page and the same backend process.

Usage:
    uv run python pipeline/app.py --project /path/to/project --video pain_test
    uv run python pipeline/app.py --project /path/to/project --video pain_test --tab masks

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


# =============================================================================
# Shell HTML — page chrome, tab switcher, shared CSS/JS; each tab module
# contributes its own HTML fragment + JS namespace.
# =============================================================================

def _build_shell_html() -> str:
    video_name   = APP_STATE["video_name"]
    initial_tab  = APP_STATE["initial_tab"]

    background_fragment = build_background_tab_html(video_name)
    masks_fragment       = build_masks_tab_html(video_name)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline — {video_name}</title>
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
  <h1>{video_name}</h1>
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
const VIDEO_NAME = "{video_name}";
let activeTab = "{initial_tab}";

function switchTab(tab) {{
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("pane-" + tab).classList.add("active");
  document.querySelector(`.tab-btn[data-tab="${{tab}}"]`).classList.add("active");
  activeTab = tab;
  if (window.onTabActivated) window.onTabActivated(tab);
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
  switchTab(activeTab);
  listenForServerShutdown();
}};
</script>
</body>
</html>"""


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global _server

    parser = argparse.ArgumentParser(description="Unified pipeline app.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--video",   required=True)
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

    video_file = project_dir / "1_videos" / f"{args.video}.{video_extension}"
    if not video_file.exists():
        logger.error("Video not found: %s", video_file)
        sys.exit(1)

    masks_dir = project_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    pv_dir = project_dir / "2_pv"
    pv_dir.mkdir(exist_ok=True)

    capture      = cv2.VideoCapture(str(video_file))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()

    display_frame_index = args.frame if args.frame is not None else total_frames // 2
    capture = cv2.VideoCapture(str(video_file))
    capture.set(cv2.CAP_PROP_POS_FRAMES, display_frame_index)
    success, display_frame = capture.read()
    capture.release()
    if not success:
        logger.error("Could not read frame %d from %s", display_frame_index, video_file)
        sys.exit(1)

    # Populate shared app state — read by all tab modules
    APP_STATE["project_dir"]      = project_dir
    APP_STATE["project_yaml_file"] = project_yaml_file
    APP_STATE["video_name"]       = args.video
    APP_STATE["video_file"]       = video_file
    APP_STATE["video_extension"]  = video_extension
    APP_STATE["meta_real_width"]  = meta_real_width
    APP_STATE["masks_dir"]        = masks_dir
    APP_STATE["pv_dir"]           = pv_dir
    APP_STATE["fps"]              = fps
    APP_STATE["total_frames"]     = total_frames
    APP_STATE["display_frame"]    = display_frame   # used by masks tab
    APP_STATE["initial_tab"]      = args.tab

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
