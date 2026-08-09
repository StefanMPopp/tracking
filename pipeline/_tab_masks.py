"""
_tab_masks.py — Masks tab for the unified pipeline app.

The mask editor is complex enough (canvas drawing, drag/rotate/resize state)
that it is kept as a fully self-contained HTML document, served at
/masks-frame and embedded via <iframe> in the masks tab pane. This avoids any
risk of JS variable/function collisions with other tabs while still giving
a uniform shell (tab switcher, close button) around it.

The underlying editor logic and routes are unchanged from the original
standalone masks.py — only the entry point (served via iframe instead of as
the page root) and route registration (via register_masks_routes(app, state)
instead of module-level @app decorators) differ. Route paths are prefixed
with /masks-frame/ to avoid colliding with other tabs' routes.
"""

import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from _masks import (
    delete_polygon,
    detect_circles,
    load_default_masks,
    load_masks_for_video,
    save_batch_masks,
    save_default_masks,
    save_polygon,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic models
# =============================================================================

class PolygonSaveRequest(BaseModel):
    vertices:  list[list[float]]
    mask_type: str
    prefix:    str

class DefaultSaveRequest(BaseModel):
    shapes: list[dict]

class DetectRequest(BaseModel):
    diameter_cm:    float
    thickness_cm:   float
    hue_center:     int
    hue_tolerance:  int
    saturation_min: int = 80
    value_max:      int = 120
    expected_count: int = 1

class AutoDetectParamsRequest(BaseModel):
    params: dict


# =============================================================================
# Route registration
# =============================================================================

def register_masks_routes(app: FastAPI, state: dict) -> None:

    @app.get("/masks-frame/frame")
    def get_frame():
        frame     = state["display_frame"]
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64       = base64.b64encode(buffer).decode("utf-8")
        return JSONResponse({"image": b64, "frame_width": frame.shape[1], "frame_height": frame.shape[0]})

    @app.get("/masks-frame/masks")
    def get_masks():
        masks_dir  = state["masks_dir"]
        video_name = state["video_name"]
        shapes = []
        for mask_file in (sorted(masks_dir.glob(f"{video_name}_include_*.csv")) +
                          sorted(masks_dir.glob(f"{video_name}_ignore_*.csv"))):
            mask_type = "include" if "_include_" in mask_file.name else "ignore"
            shapes.append({"filename": mask_file.name, "type": mask_type,
                           "vertices": _read_polygon_csv_raw(mask_file)})
        return JSONResponse({"shapes": shapes})

    @app.get("/masks-frame/masks/defaults")
    def get_default_masks():
        masks_dir = state["masks_dir"]
        shapes = []
        for mask_file in sorted(masks_dir.glob("default_*.csv")):
            mask_type = "include" if "_include_" in mask_file.name else "ignore"
            shapes.append({"filename": mask_file.name, "type": mask_type,
                           "vertices": _read_polygon_csv_raw(mask_file)})
        return JSONResponse({"shapes": shapes})

    @app.get("/masks-frame/masks/batch")
    def get_batch_masks():
        """
        Return the batch containing the current video (if any) and that
        batch's mask shapes. batch_name is null if the video is not in
        any batch.
        """
        from _resolve import resolve_batch_for_video
        project_config = yaml.safe_load(state["project_yaml_file"].read_text())
        batch_name = resolve_batch_for_video(state["video_name"], project_config)

        shapes = []
        if batch_name is not None:
            masks_dir = state["masks_dir"]
            for mask_file in sorted(masks_dir.glob(f"{batch_name}_include_*.csv")) + \
                             sorted(masks_dir.glob(f"{batch_name}_ignore_*.csv")):
                mask_type = "include" if "_include_" in mask_file.name else "ignore"
                shapes.append({"filename": mask_file.name, "type": mask_type,
                               "vertices": _read_polygon_csv_raw(mask_file)})

        return JSONResponse({"batch_name": batch_name, "shapes": shapes})

    @app.post("/masks-frame/masks/save-batch-defaults")
    def save_batch_defaults(request: DefaultSaveRequest):
        """Save current shapes as defaults for the batch containing this video."""
        from _resolve import resolve_batch_for_video
        project_config = yaml.safe_load(state["project_yaml_file"].read_text())
        batch_name = resolve_batch_for_video(state["video_name"], project_config)
        if batch_name is None:
            raise HTTPException(
                status_code=400,
                detail="Current video does not belong to any batch.",
            )
        save_batch_masks(request.shapes, batch_name, state["masks_dir"])
        return JSONResponse({"saved": len(request.shapes), "batch_name": batch_name})

    @app.post("/masks-frame/masks/save")
    def save_mask(request: PolygonSaveRequest):
        masks_dir = state["masks_dir"]
        if request.mask_type not in ("include", "ignore"):
            raise HTTPException(status_code=400, detail="mask_type must be 'include' or 'ignore'")
        filename = save_polygon(polygon=request.vertices, prefix=request.prefix,
                                mask_type=request.mask_type, masks_dir=masks_dir)
        return JSONResponse({"filename": filename})

    @app.delete("/masks-frame/masks/{filename}")
    def delete_mask(filename: str):
        delete_polygon(filename, state["masks_dir"])
        return JSONResponse({"deleted": filename})

    @app.post("/masks-frame/masks/save-defaults")
    def save_defaults(request: DefaultSaveRequest):
        save_default_masks(request.shapes, state["masks_dir"])
        return JSONResponse({"saved": len(request.shapes)})

    @app.post("/masks-frame/masks/detect")
    def run_detection(request: DetectRequest):
        frame    = state["display_frame"]
        polygons = detect_circles(
            frame=frame, diameter_cm=request.diameter_cm, thickness_cm=request.thickness_cm,
            meta_real_width=state["meta_real_width"], expected_count=request.expected_count,
            hue_center=request.hue_center, hue_tolerance=request.hue_tolerance,
            saturation_min=request.saturation_min, value_max=request.value_max,
        )
        return JSONResponse({"polygons": polygons, "count": len(polygons)})

    @app.post("/masks-frame/masks/detect-debug")
    def run_detection_debug(request: DetectRequest):
        """
        Same parameters as /masks/detect, but returns the intermediate HSV
        mask, the post-cleanup mask, and an annotated contours image showing
        why each candidate blob was accepted or rejected — for diagnosing
        "nothing detected" cases.
        """
        from _masks import detect_circles_debug
        frame  = state["display_frame"]
        result = detect_circles_debug(
            frame=frame, diameter_cm=request.diameter_cm, thickness_cm=request.thickness_cm,
            meta_real_width=state["meta_real_width"], expected_count=request.expected_count,
            hue_center=request.hue_center, hue_tolerance=request.hue_tolerance,
            saturation_min=request.saturation_min, value_max=request.value_max,
        )
        return JSONResponse(result)

    @app.get("/masks-frame/project/auto-detect-params")
    def get_auto_detect_params():
        cfg = yaml.safe_load(state["project_yaml_file"].read_text())
        return JSONResponse(cfg.get("auto_detect_circles") or {})

    @app.post("/masks-frame/project/auto-detect-params")
    def save_auto_detect_params(request: AutoDetectParamsRequest):
        f   = state["project_yaml_file"]
        cfg = yaml.safe_load(f.read_text())
        cfg["auto_detect_circles"] = request.params
        f.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        return JSONResponse({"saved": True})

    @app.get("/masks-frame", response_class=HTMLResponse)
    def masks_frame():
        frame = state["display_frame"]
        return HTMLResponse(_build_masks_frame_html(
            video_name=state["video_name"],
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        ))


# =============================================================================
# HTML fragment for the tab (just an iframe pointing at /masks-frame)
# =============================================================================

def build_masks_tab_html(video_name: str) -> str:
    return """
<iframe id="masks-iframe" src="/masks-frame" style="width:100%; height:100%; border:none;"></iframe>
"""


# =============================================================================
# Full standalone HTML document for the iframe (original masks editor,
# routes repointed to /masks-frame/* to avoid colliding with other tabs)
# =============================================================================

def _build_masks_frame_html(video_name: str, frame_width: int, frame_height: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mask Editor — {video_name}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0;
       display: flex; height: 100vh; overflow: hidden; }}

#sidebar {{
  width: 260px; min-width: 260px; background: #16213e;
  display: flex; flex-direction: column; padding: 12px; gap: 10px;
  overflow-y: auto; border-right: 1px solid #0f3460;
}}
h2 {{ font-size: 13px; color: #a0c4ff; text-transform: uppercase; letter-spacing: 1px; }}
.section {{ background: #0f3460; border-radius: 6px; padding: 10px; }}
.section h3 {{ font-size: 11px; color: #7ec8e3; text-transform: uppercase;
              letter-spacing: 0.8px; margin-bottom: 8px; }}
button {{ width: 100%; padding: 7px 10px; border: none; border-radius: 5px;
          cursor: pointer; font-size: 12px; font-weight: 600; transition: opacity .15s; }}
button:hover {{ opacity: 0.85; }}
button.active {{ outline: 2px solid #fff; }}
.btn-include  {{ background: #2d6a4f; color: #d8f3dc; }}
.btn-ignore   {{ background: #9d0208; color: #ffccd5; }}
.btn-neutral  {{ background: #415a77; color: #e0e0e0; }}
.btn-detect   {{ background: #e76f51; color: #fff; }}
.btn-default  {{ background: #5c4033; color: #ffe0b2; }}
.btn-save-def {{ background: #6a0572; color: #f3e5f5; }}
.btn-close    {{ background: #7b2d00; color: #ffe0d0; }}
.mode-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }}
label {{ font-size: 11px; color: #aaa; display: block; margin-bottom: 2px; }}
input[type=number], input[type=range] {{
  width: 100%; padding: 4px 6px; background: #1a1a2e; border: 1px solid #415a77;
  border-radius: 4px; color: #e0e0e0; font-size: 12px; }}
input[type=color] {{
  width: 100%; height: 30px; padding: 2px; background: #1a1a2e;
  border: 1px solid #415a77; border-radius: 4px; cursor: pointer; }}
.field {{ margin-bottom: 6px; }}
#status {{ font-size: 11px; color: #7ec8e3; padding: 6px; background: #0f3460;
           border-radius: 4px; min-height: 32px; }}
#shape-list {{ max-height: 200px; overflow-y: auto; }}
.shape-item {{ display: flex; justify-content: space-between; align-items: center;
               padding: 4px 6px; border-radius: 4px; margin-bottom: 3px;
               font-size: 11px; cursor: pointer; }}
.shape-item:hover {{ background: #1a3a5c; }}
.shape-item.selected {{ background: #1a3a5c; outline: 1px solid #a0c4ff; }}
.shape-item .dot {{ width: 10px; height: 10px; border-radius: 50%;
                    flex-shrink: 0; margin-right: 6px; }}
.shape-item .del-btn {{ background: none; border: none; color: #ff6b6b;
                        cursor: pointer; font-size: 13px; width: auto; padding: 0 4px; }}
#type-bar {{ display: flex; gap: 6px; }}
#type-bar button {{ flex: 1; }}
#canvas-wrap {{ flex: 1; overflow: auto; padding: 10px; }}
canvas {{ display: block; margin: auto; }}
#shutdown-overlay {{ display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.75); align-items: center;
  justify-content: center; z-index: 100; }}
#shutdown-overlay.visible {{ display: flex; }}
#shutdown-box {{ background: #16213e; border: 1px solid #a0c4ff; border-radius: 10px;
  padding: 28px 32px; text-align: center; max-width: 340px; }}
#shutdown-box h3 {{ color: #a0c4ff; margin-bottom: 14px; }}
#shutdown-box p  {{ color: #ccc; font-size: 13px; margin-bottom: 20px; }}
.btn-row {{ display: flex; gap: 10px; justify-content: center; }}
.btn-row button {{ width: auto; padding: 8px 20px; }}
#debug-overlay {{ display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.9); align-items: center;
  justify-content: center; z-index: 110; overflow-y: auto; padding: 20px; }}
#debug-overlay.visible {{ display: flex; }}
#debug-box {{ background: #16213e; border: 1px solid #a0c4ff; border-radius: 10px;
  padding: 20px; max-width: 95vw; max-height: 95vh; overflow-y: auto; }}
#debug-box h3 {{ color: #a0c4ff; margin-bottom: 12px; }}
#debug-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }}
.debug-cell {{ background: #000; border-radius: 6px; overflow: hidden; }}
.debug-cell img {{ width: 100%; display: block; }}
.debug-cell-label {{ font-size: 11px; color: #7ec8e3; padding: 4px 8px; background: #0f3460; }}
#debug-legend {{ font-size: 11px; color: #ccc; margin-bottom: 10px; line-height: 1.6; }}
#debug-contour-table {{ width: 100%; font-size: 11px; border-collapse: collapse; margin-bottom: 14px; }}
#debug-contour-table th, #debug-contour-table td {{ padding: 4px 8px; text-align: left;
  border-bottom: 1px solid #0f3460; }}
#debug-contour-table th {{ color: #7ec8e3; }}
.debug-accepted {{ color: #50dc50; }}
.debug-rejected {{ color: #dc5050; }}
</style>
</head>
<body>

<div id="sidebar">
  <h2>Mask Editor</h2>
  <div id="status">Loading…</div>
  <div style="display:flex; align-items:center; gap:6px; font-size:11px; color:#7ec8e3;">
    <span>Zoom: <span id="zoom-level-label">100%</span></span>
    <button class="btn-neutral" style="width:auto; padding:2px 8px;" onclick="resetZoom()">Reset</button>
    <span style="color:#5a7a99;">(Ctrl+scroll to zoom)</span>
  </div>

  <div class="section">
    <h3>Draw mode</h3>
    <div class="mode-grid">
      <button class="btn-neutral" id="btn-perimeter" onclick="setMode('perimeter')">✏️ Perimeter</button>
      <button class="btn-neutral" id="btn-rectangle" onclick="setMode('rectangle')">▭ Rectangle</button>
      <button class="btn-neutral" id="btn-oval"      onclick="setMode('oval')">⬭ Oval</button>
      <button class="btn-neutral" id="btn-select"    onclick="setMode('select')">↖ Select</button>
    </div>
  </div>

  <div class="section">
    <h3>Mask type</h3>
    <div id="type-bar">
      <button class="btn-include" id="btn-type-include" onclick="setMaskType('include')">✅ Include</button>
      <button class="btn-ignore"  id="btn-type-ignore"  onclick="setMaskType('ignore')">🚫 Ignore</button>
    </div>
  </div>

  <div class="section">
    <h3>Auto-detect circles</h3>
    <div class="field"><label>Diameter (cm)</label>
      <input type="number" id="det-diameter" step="0.1" value="4.5"></div>
    <div class="field"><label>Ring thickness (cm)</label>
      <input type="number" id="det-thickness" step="0.1" value="0.3"></div>
    <div class="field"><label>Expected count</label>
      <input type="number" id="det-count" min="1" value="1"></div>
    <div class="field"><label>Ring colour</label>
      <input type="color" id="det-color" value="#8b0000"></div>
    <div class="field"><label>Hue tolerance (±)</label>
      <input type="range" id="det-hue-tol" min="5" max="40" value="15">
      <span id="det-hue-tol-val" style="font-size:11px">15</span></div>
    <div class="field"><label>Max brightness (0=dark)</label>
      <input type="range" id="det-brightness" min="20" max="200" value="120">
      <span id="det-brightness-val" style="font-size:11px">120</span></div>
    <button class="btn-detect"  onclick="runDetect()">🔍 Detect</button>
    <button class="btn-neutral" style="margin-top:5px" onclick="runDetectDebug()">🩺 Debug detection</button>
    <button class="btn-neutral" style="margin-top:5px" onclick="saveDetectParams()">💾 Save params to project</button>
  </div>

  <div class="section">
    <h3>Batch defaults</h3>
    <div id="batch-name-label" style="font-size:11px; color:#789; margin-bottom:6px;">
      Checking batch…</div>
    <button class="btn-default"  onclick="loadBatchDefaults()">📂 Load batch defaults</button>
    <button class="btn-save-def" style="margin-top:5px" onclick="saveAsBatchDefaults()">⭐ Save current as batch defaults</button>
  </div>

  <div class="section">
    <h3>Project defaults</h3>
    <button class="btn-default"  onclick="loadDefaults()">📂 Load project defaults</button>
    <button class="btn-save-def" style="margin-top:5px"
            title="Ctrl+Shift+S" onclick="saveAsDefaults()">⭐ Save current as defaults</button>
  </div>

  <div class="section">
    <h3>Shapes (<span id="shape-count">0</span>)</h3>
    <div id="shape-list"></div>
  </div>

  <button class="btn-neutral" title="Ctrl+S" onclick="saveAll()"
          style="background:#1b4332;color:#d8f3dc">💾 Save all unsaved (Ctrl+S)</button>
  <button class="btn-close" onclick="requestClose()">✖ Close editor (Q)</button>
</div>

<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>

<div id="shutdown-overlay">
  <div id="shutdown-box">
    <h3>Close editor?</h3>
    <p id="shutdown-msg">You have unsaved shapes.</p>
    <div class="btn-row">
      <button class="btn-include" id="btn-save-close" onclick="saveAndClose()">💾 Save &amp; close</button>
      <button class="btn-ignore"  onclick="closeNow()">✖ Close without saving</button>
      <button class="btn-neutral" onclick="cancelClose()">Cancel</button>
    </div>
  </div>
</div>

<div id="debug-overlay">
  <div id="debug-box">
    <h3>🩺 Detection debug</h3>
    <div id="debug-legend">
      <strong>Reading this:</strong> the colour mask shows every pixel that passed the hue/saturation/brightness
      test — gaps or holes here mean lighting unevenness knocked those pixels out of range.
      The cleaned mask shows the same after morphological closing/opening tries to patch small gaps.
      The contours image draws every blob OpenCV found on the cleaned mask:
      <span class="debug-accepted">green = accepted</span>,
      <span class="debug-rejected">red = rejected</span> (with its measured area/circularity labelled).
      Expected area range for your current parameters: <span id="debug-area-range"></span>.
    </div>
    <div id="debug-grid"></div>
    <table id="debug-contour-table">
      <thead><tr><th>#</th><th>Area</th><th>Circularity</th><th>Result</th></tr></thead>
      <tbody id="debug-contour-rows"></tbody>
    </table>
    <div class="btn-row">
      <button class="btn-neutral" onclick="closeDebugOverlay()">Close</button>
    </div>
  </div>
</div>

<script>
// ============================================================
// Constants
// ============================================================
const HANDLE_RADIUS   = 4;
const ROT_HANDLE_DIST = 22;
const BB_HANDLE_RADIUS = 5;   // bounding-box corner/edge handles
const INCLUDE_FILL    = "rgba(45,180,100,0.22)";
const INCLUDE_STROKE  = "rgba(45,220,80,0.9)";
const IGNORE_FILL     = "rgba(220,30,30,0.22)";
const IGNORE_STROKE   = "rgba(220,50,50,0.9)";
const PROPOSED_FILL   = "rgba(255,200,0,0.15)";
const PROPOSED_STROKE = "rgba(255,200,0,0.85)";
const BB_STROKE       = "rgba(180,210,255,0.7)";
const VIDEO_NAME      = "{video_name}";

// ============================================================
// State
// ============================================================
let mode       = "perimeter";
let maskType   = "include";
let shapes     = [];
let selectedId = null;
let nextId     = 1;

let perimeterVerts = [];
let dragStart      = null;
let dragCurrent    = null;

// Unified drag state for select mode
// dragOp: null | "move" | "vertex" | "rotate" | "bb" | "copy"
let dragOp          = null;
let dragShapeId     = null;
let dragVertexIdx   = null;   // for "vertex" op
let dragBbHandle    = null;   // 0-7 for "bb" op (corners TL,TR,BR,BL + edges T,R,B,L)
let dragOffset      = null;   // [dx,dy] display coords at drag start
let shapeStartVerts = null;
let shapeBbAtDrag   = null;   // {{x,y,w,h}} at drag start for bb scaling
let rotCenterAtDrag = null;
let rotStartAngle   = null;

let canvas, ctx, img;
let displayScale = 1;
let baseFitScale  = 1;   // the "fit to wrap" scale before zoom is applied
let zoomFactor    = 1;   // multiplier on top of baseFitScale, via Ctrl+wheel
let nativeW, nativeH;

const ZOOM_MIN = 0.25;
const ZOOM_MAX = 6;
const ZOOM_STEP = 0.1;

// ============================================================
// Init
// ============================================================
window.onload = async () => {{
  canvas = document.getElementById("canvas");
  ctx    = canvas.getContext("2d");

  canvas.addEventListener("mousedown",   onMouseDown);
  canvas.addEventListener("mousemove",   onMouseMove);
  canvas.addEventListener("mouseup",     onMouseUp);
  canvas.addEventListener("dblclick",    onDblClick);
  canvas.addEventListener("contextmenu", onRightClick);
  document.getElementById("canvas-wrap").addEventListener("wheel", onCanvasWheel, {{passive: false}});

  setMode("perimeter");
  setMaskType("include");
  wireSliders();
  await loadFrame();
  await loadExistingMasks();
  await loadAutoDetectParams();
  await checkBatch();
  listenForServerShutdown();
  setStatus("Ready — " + VIDEO_NAME);
}};

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

function wireSliders() {{
  [["det-hue-tol","det-hue-tol-val"],["det-brightness","det-brightness-val"]].forEach(([s,v]) => {{
    const el = document.getElementById(s), vl = document.getElementById(v);
    el.oninput = () => vl.textContent = el.value;
  }});
}}

async function loadFrame() {{
  const res = await fetch("/masks-frame/frame"), d = await res.json();
  nativeW = d.frame_width; nativeH = d.frame_height;
  img = new Image();
  img.src = "data:image/jpeg;base64," + d.image;
  await new Promise(r => img.onload = r);
  resizeCanvas();
  window.onresize = resizeCanvas;
}}

function resizeCanvas() {{
  const wrap = document.getElementById("canvas-wrap");
  const maxW = wrap.clientWidth - 10, maxH = wrap.clientHeight - 10;
  baseFitScale  = Math.min(maxW / nativeW, maxH / nativeH, 1);
  displayScale  = baseFitScale * zoomFactor;
  canvas.width  = Math.round(nativeW * displayScale);
  canvas.height = Math.round(nativeH * displayScale);
  canvas.style.cursor = (mode === "select") ? "default" : "crosshair";
  redraw();
}}

function onCanvasWheel(e) {{
  if (!e.ctrlKey) return;   // only intercept Ctrl+wheel; let normal scroll pass through
  e.preventDefault();

  const wrap = document.getElementById("canvas-wrap");
  const rectBefore = canvas.getBoundingClientRect();
  // Cursor position relative to the canvas, in pre-zoom canvas pixels
  const cursorX = e.clientX - rectBefore.left;
  const cursorY = e.clientY - rectBefore.top;
  // Same position in native (unscaled) image coordinates — stays fixed under the cursor
  const [nativeX, nativeY] = toNative(cursorX, cursorY);

  const delta = e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
  zoomFactor = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoomFactor + delta));

  resizeCanvas();
  updateZoomLabel();

  // Re-scroll the wrap so the same native point stays under the cursor
  const [newCursorX, newCursorY] = toDisplay(nativeX, nativeY);
  wrap.scrollLeft += (newCursorX - cursorX);
  wrap.scrollTop  += (newCursorY - cursorY);
}}

function updateZoomLabel() {{
  document.getElementById("zoom-level-label").textContent =
    Math.round(zoomFactor * 100) + "%";
}}

function resetZoom() {{
  zoomFactor = 1;
  resizeCanvas();
  updateZoomLabel();
}}

async function loadExistingMasks() {{
  const res = await fetch("/masks-frame/masks"), d = await res.json();
  d.shapes.forEach(s => addShape(s.vertices, s.type, s.filename));
}}

async function loadAutoDetectParams() {{
  const res = await fetch("/masks-frame/project/auto-detect-params"), d = await res.json();
  if (d.diameter_cm)    document.getElementById("det-diameter").value  = d.diameter_cm;
  if (d.thickness_cm)   document.getElementById("det-thickness").value = d.thickness_cm;
  if (d.expected_count) document.getElementById("det-count").value     = d.expected_count;
}}

// ============================================================
// Shape management
// ============================================================
function addShape(vertices, type, filename=null, proposed=false) {{
  const id = nextId++;
  shapes.push({{ id, type, vertices: vertices.map(v=>[...v]), filename, proposed }});
  selectedId = id;
  refreshShapeList(); redraw();
  return id;
}}

function removeShape(id) {{
  const shape = shapes.find(s => s.id === id);
  if (!shape) return;
  if (shape.filename) fetch(`/masks-frame/masks/${{shape.filename}}`, {{method:"DELETE"}});
  shapes    = shapes.filter(s => s.id !== id);
  selectedId = (selectedId === id) ? null : selectedId;
  refreshShapeList(); redraw();
}}

function refreshShapeList() {{
  const list = document.getElementById("shape-list");
  document.getElementById("shape-count").textContent = shapes.length;
  list.innerHTML = "";
  shapes.forEach(s => {{
    const div = document.createElement("div");
    div.className = "shape-item" + (s.id === selectedId ? " selected" : "");
    div.onclick   = () => {{ selectedId = s.id; refreshShapeList(); redraw(); }};
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = s.proposed ? "#ffd700" : s.type==="include" ? "#2db864" : "#dc1e1e";
    const lbl = document.createElement("span");
    lbl.style.flex = "1";
    lbl.textContent = (s.filename || (s.proposed?"proposed":"unsaved")) + " (" + s.vertices.length + "v)";
    const del = document.createElement("button");
    del.className = "del-btn"; del.textContent = "✕";
    del.onclick = e => {{ e.stopPropagation(); removeShape(s.id); }};
    div.append(dot, lbl, del); list.appendChild(div);
  }});
}}

function hasUnsaved() {{ return shapes.some(s => !s.filename); }}

// ============================================================
// Mode / type
// ============================================================
function setMode(m) {{
  mode = m; perimeterVerts = []; dragStart = dragCurrent = null; dragOp = null;
  ["perimeter","rectangle","oval","select"].forEach(n =>
    document.getElementById("btn-"+n).classList.toggle("active", n===m));
  if (canvas) canvas.style.cursor = (m==="select") ? "default" : "crosshair";
  redraw();
}}

function setMaskType(t) {{
  maskType = t;
  document.getElementById("btn-type-include").classList.toggle("active", t==="include");
  document.getElementById("btn-type-ignore" ).classList.toggle("active", t==="ignore");
}}

// ============================================================
// Coordinate helpers
// ============================================================
function toNative(dx, dy)  {{ return [dx / displayScale, dy / displayScale]; }}
function toDisplay(nx, ny) {{ return [nx * displayScale, ny * displayScale]; }}
function canvasXY(e) {{
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}}

// ============================================================
// Bounding box helpers
// ============================================================
function shapeBoundingBox(verts) {{
  const xs = verts.map(v=>v[0]), ys = verts.map(v=>v[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  return {{ x, y, w: Math.max(...xs)-x, h: Math.max(...ys)-y }};
}}

// Returns 8 handle positions [x,y] in NATIVE coords:
// 0=TL 1=TR 2=BR 3=BL 4=TC 5=RC 6=BC 7=LC
function bbHandles(bb) {{
  const {{x,y,w,h}} = bb;
  return [
    [x,     y    ], [x+w,   y    ], [x+w,   y+h  ], [x,     y+h  ],
    [x+w/2, y    ], [x+w,   y+h/2], [x+w/2, y+h  ], [x,     y+h/2],
  ];
}}

// Appropriate cursor per bb handle index
const BB_CURSORS = ["nw-resize","ne-resize","se-resize","sw-resize",
                    "n-resize","e-resize","s-resize","w-resize"];

function hitTestBbHandles(dx, dy, shape) {{
  const bb = shapeBoundingBox(shape.vertices);
  const handles = bbHandles(bb);
  for (let i = 0; i < handles.length; i++) {{
    const [hx,hy] = toDisplay(...handles[i]);
    if (Math.hypot(dx-hx, dy-hy) < BB_HANDLE_RADIUS + 3) return i;
  }}
  return -1;
}}

function scaledVertices(startVerts, startBb, newX, newY, newW, newH) {{
  // Scale all vertices relative to the bounding box origin
  const {{x:ox, y:oy, w:ow, h:oh}} = startBb;
  return startVerts.map(([vx,vy]) => [
    newX + (ow > 0 ? (vx - ox) / ow * newW : 0),
    newY + (oh > 0 ? (vy - oy) / oh * newH : 0),
  ]);
}}

// ============================================================
// Mouse events
// ============================================================
function onMouseDown(e) {{
  e.preventDefault();
  if (e.button !== 0) return;
  const [dx, dy] = canvasXY(e);
  const [nx, ny] = toNative(dx, dy);

  if (mode === "select") {{
    const sel = shapes.find(s => s.id === selectedId);

    // 1. Bounding-box handle on selected shape?
    if (sel) {{
      const bbIdx = hitTestBbHandles(dx, dy, sel);
      if (bbIdx >= 0) {{
        dragOp       = "bb";
        dragShapeId  = sel.id;
        dragBbHandle = bbIdx;
        dragOffset   = [dx, dy];
        shapeStartVerts = sel.vertices.map(v=>[...v]);
        shapeBbAtDrag   = shapeBoundingBox(sel.vertices);
        return;
      }}
      // 2. Rotation handle?
      const [rhx,rhy] = rotHandlePos(sel);
      if (Math.hypot(dx-rhx, dy-rhy) < HANDLE_RADIUS+3) {{
        dragOp          = "rotate";
        dragShapeId     = sel.id;
        dragOffset      = [dx, dy];
        shapeStartVerts = sel.vertices.map(v=>[...v]);
        rotCenterAtDrag = centroid(sel.vertices);
        rotStartAngle   = Math.atan2(dy - rotCenterAtDrag[1]*displayScale,
                                      dx - rotCenterAtDrag[0]*displayScale);
        return;
      }}
      // 3. Vertex handle on selected shape?
      for (let i=0; i<sel.vertices.length; i++) {{
        const [vdx,vdy] = toDisplay(...sel.vertices[i]);
        if (Math.hypot(dx-vdx, dy-vdy) < HANDLE_RADIUS+3) {{
          dragOp        = "vertex";
          dragShapeId   = sel.id;
          dragVertexIdx = i;
          dragOffset    = [dx, dy];
          shapeStartVerts = sel.vertices.map(v=>[...v]);
          return;
        }}
      }}
    }}

    // 4. Hit test any shape (interior or vertex)
    const hit = hitTestShape(dx, dy);
    if (hit !== null) {{
      selectedId      = hit;
      const shape     = shapes.find(s => s.id === hit);
      shapeStartVerts = shape.vertices.map(v=>[...v]);
      dragOffset      = [dx, dy];

      if (e.ctrlKey) {{
        // Ctrl+drag = copy
        dragOp      = "copy";
        dragShapeId = hit;
      }} else {{
        dragOp      = "move";
        dragShapeId = hit;
      }}
      refreshShapeList(); redraw();
    }} else {{
      selectedId = null; dragOp = null;
      refreshShapeList(); redraw();
    }}
    return;
  }}

  if (mode === "perimeter") {{
    if (perimeterVerts.length >= 3) {{
      const [fx,fy] = toDisplay(...perimeterVerts[0]);
      if (Math.hypot(dx-fx, dy-fy) < HANDLE_RADIUS*2.5) {{ closePerimeter(); return; }}
    }}
    perimeterVerts.push([nx, ny]);
    redraw(); return;
  }}

  if (mode === "rectangle" || mode === "oval") {{
    dragStart = [nx, ny]; dragCurrent = [nx, ny]; return;
  }}
}}

function onMouseMove(e) {{
  const [dx, dy] = canvasXY(e);
  const [nx, ny] = toNative(dx, dy);

  // Update cursor when hovering over bb handles
  if (mode === "select" && !dragOp) {{
    const sel = shapes.find(s => s.id === selectedId);
    if (sel) {{
      const bbIdx = hitTestBbHandles(dx, dy, sel);
      if (bbIdx >= 0) {{ canvas.style.cursor = BB_CURSORS[bbIdx]; redraw(); return; }}
    }}
    canvas.style.cursor = hitTestShape(dx, dy) !== null ? "move" : "default";
  }}

  if ((mode==="rectangle"||mode==="oval") && dragStart) {{
    dragCurrent = [nx, ny]; redraw(); return;
  }}

  if (mode==="select" && dragOp) {{
    const shape = shapes.find(s => s.id === dragShapeId);
    if (!shape) return;
    const ddx = (dx - dragOffset[0]) / displayScale;
    const ddy = (dy - dragOffset[1]) / displayScale;

    if (dragOp === "rotate") {{
      const [cx,cy]  = rotCenterAtDrag;
      const angle    = Math.atan2(dy - cy*displayScale, dx - cx*displayScale);
      const delta    = angle - rotStartAngle;
      shape.vertices = shapeStartVerts.map(([vx,vy]) => {{
        const rx=vx-cx, ry=vy-cy;
        return [cx+rx*Math.cos(delta)-ry*Math.sin(delta),
                cy+rx*Math.sin(delta)+ry*Math.cos(delta)];
      }});

    }} else if (dragOp === "vertex") {{
      const sv = shapeStartVerts[dragVertexIdx];
      shape.vertices[dragVertexIdx] = [sv[0]+ddx, sv[1]+ddy];

    }} else if (dragOp === "move" || dragOp === "copy") {{
      shape.vertices = shapeStartVerts.map(([vx,vy]) => [vx+ddx, vy+ddy]);

    }} else if (dragOp === "bb") {{
      const {{x:ox,y:oy,w:ow,h:oh}} = shapeBbAtDrag;
      let [nx2, ny2] = toNative(dx, dy);
      let newX=ox, newY=oy, newW=ow, newH=oh;

      // Which edges does this handle affect?
      //   TL=0: left+top  TR=1: right+top  BR=2: right+bot  BL=3: left+bot
      //   TC=4: top        RC=5: right       BC=6: bot         LC=7: left
      const movesLeft  = [0,3,7].includes(dragBbHandle);
      const movesRight = [1,2,5].includes(dragBbHandle);
      const movesTop   = [0,1,4].includes(dragBbHandle);
      const movesBott  = [2,3,6].includes(dragBbHandle);

      if (movesRight)  newW = Math.max(4, nx2 - ox);
      if (movesLeft)  {{ newX = Math.min(nx2, ox+ow-4); newW = ow + (ox-newX); }}
      if (movesBott)   newH = Math.max(4, ny2 - oy);
      if (movesTop)   {{ newY = Math.min(ny2, oy+oh-4); newH = oh + (oy-newY); }}

      // Shift+drag: preserve aspect ratio
      if (e.shiftKey && (movesLeft||movesRight) && (movesTop||movesBott)) {{
        const aspect = ow/oh;
        const dw = newW-ow, dh = newH-oh;
        if (Math.abs(dw/ow) > Math.abs(dh/oh)) newH = newW/aspect;
        else newW = newH*aspect;
        // Re-anchor the fixed corner
        if (movesLeft)  newX = (ox+ow) - newW;
        if (movesTop)   newY = (oy+oh) - newH;
      }}

      shape.vertices = scaledVertices(shapeStartVerts, shapeBbAtDrag, newX, newY, newW, newH);
    }}
    shape.filename = null;   // mark modified as unsaved
    redraw(); return;
  }}

  if (mode==="perimeter" && perimeterVerts.length>0) {{
    redraw();
    const [lx,ly]=toDisplay(...perimeterVerts[perimeterVerts.length-1]);
    ctx.strokeStyle="#ffd700"; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(dx,dy); ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function onMouseUp(e) {{
  if (mode==="rectangle"||mode==="oval") {{
    if (!dragStart||!dragCurrent) return;
    const [x0,y0]=dragStart,[x1,y1]=dragCurrent;
    if (Math.abs(x1-x0)<3&&Math.abs(y1-y0)<3) {{ dragStart=dragCurrent=null; return; }}
    const verts = mode==="rectangle"
      ? [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
      : ovalVertices(x0,y0,x1,y1,32);
    addShape(verts, maskType);
    dragStart=dragCurrent=null; return;
  }}

  if (mode==="select" && dragOp) {{
    if (dragOp === "copy") {{
      // Finalise the copy: the shape currently has the moved position (preview).
      // Restore original to original position, create new shape at moved position.
      const shape   = shapes.find(s => s.id === dragShapeId);
      const movedVerts = shape.vertices.map(v=>[...v]);
      shape.vertices = shapeStartVerts;   // restore original
      addShape(movedVerts, shape.type);   // add copy (unsaved)
    }}
    dragOp = null; redraw();
  }}
}}

function onDblClick(e) {{
  if (mode==="perimeter" && perimeterVerts.length>=3) closePerimeter();
}}

function onRightClick(e) {{
  e.preventDefault();
  const [dx,dy] = canvasXY(e);

  if (mode==="select") {{
    const sel = shapes.find(s => s.id === selectedId);
    if (sel) {{
      // Right-click on a vertex handle?
      for (let i=0; i<sel.vertices.length; i++) {{
        const [vdx,vdy]=toDisplay(...sel.vertices[i]);
        if (Math.hypot(dx-vdx,dy-vdy)<HANDLE_RADIUS+3) {{
          if (sel.vertices.length > 3) {{
            sel.vertices.splice(i,1); sel.filename=null; redraw();
          }}
          return;
        }}
      }}
    }}
    const hit = hitTestShape(dx, dy);
    if (hit !== null) removeShape(hit);
    return;
  }}
  if (mode==="perimeter" && perimeterVerts.length>0) {{
    perimeterVerts.pop(); redraw();
  }}
}}

// ============================================================
// Perimeter
// ============================================================
function closePerimeter() {{
  if (perimeterVerts.length>=3) addShape([...perimeterVerts], maskType);
  perimeterVerts=[]; redraw();
}}

document.addEventListener("keydown", e => {{
  if (e.target.tagName==="INPUT") return;

  // Ctrl+S / Ctrl+Shift+S
  if (e.ctrlKey && e.shiftKey && (e.key==="s"||e.key==="S")) {{
    e.preventDefault(); saveAsDefaults(); return;
  }}
  if (e.ctrlKey && (e.key==="s"||e.key==="S")) {{
    e.preventDefault(); saveAll(); return;
  }}

  if (e.key==="Enter" && document.getElementById("shutdown-overlay").classList.contains("visible")) {{
    closeNow(); return;
  }}
  if (e.key==="Enter"&&mode==="perimeter"&&perimeterVerts.length>=3) closePerimeter();
  if (e.key==="Escape") {{
    document.getElementById("debug-overlay").classList.remove("visible");
    perimeterVerts=[]; dragStart=dragCurrent=null; dragOp=null; redraw();
  }}
  if (e.key==="s"||e.key==="S") setMode("select");
  if (e.key==="p"||e.key==="P") setMode("perimeter");
  if (e.key==="r"||e.key==="R") setMode("rectangle");
  if (e.key==="o"||e.key==="O") setMode("oval");
  if (e.key==="i"||e.key==="I") setMaskType("include");
  if (e.key==="g"||e.key==="G") setMaskType("ignore");
  if ((e.key==="Delete"||e.key==="Backspace")&&selectedId) removeShape(selectedId);
  if (e.key==="q"||e.key==="Q") requestClose();
}});

// ============================================================
// Hit testing (interior only — vertex/bb handled in onMouseDown)
// ============================================================
function hitTestShape(dx, dy) {{
  for (const shape of [...shapes].reverse()) {{
    if (pointInPolygon([dx/displayScale, dy/displayScale], shape.vertices))
      return shape.id;
  }}
  return null;
}}

function pointInPolygon([px,py], verts) {{
  let inside=false;
  for (let i=0,j=verts.length-1; i<verts.length; j=i++) {{
    const [xi,yi]=verts[i],[xj,yj]=verts[j];
    if (((yi>py)!=(yj>py))&&(px<(xj-xi)*(py-yi)/(yj-yi)+xi)) inside=!inside;
  }}
  return inside;
}}

// ============================================================
// Geometry helpers
// ============================================================
function centroid(verts) {{
  return [verts.reduce((s,[x])=>s+x,0)/verts.length,
          verts.reduce((s,[,y])=>s+y,0)/verts.length];
}}

function rotHandlePos(shape) {{
  const [cx] = centroid(shape.vertices);
  const minY = Math.min(...shape.vertices.map(v=>v[1]));
  const [dcx,dcy] = toDisplay(cx,minY);
  return [dcx, dcy-ROT_HANDLE_DIST];
}}

function ovalVertices(x0,y0,x1,y1,n) {{
  const cx=(x0+x1)/2, cy=(y0+y1)/2, rx=Math.abs(x1-x0)/2, ry=Math.abs(y1-y0)/2;
  return Array.from({{length:n}},(_,i)=>[
    cx+rx*Math.cos(2*Math.PI*i/n), cy+ry*Math.sin(2*Math.PI*i/n)]);
}}

// ============================================================
// Drawing
// ============================================================
function redraw() {{
  if (!img) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
  shapes.forEach(s => drawShape(s));

  if (mode==="perimeter"&&perimeterVerts.length>0) {{
    const stroke = maskType==="include" ? INCLUDE_STROKE : IGNORE_STROKE;
    ctx.strokeStyle=stroke; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
    ctx.beginPath();
    const [fx,fy]=toDisplay(...perimeterVerts[0]); ctx.moveTo(fx,fy);
    perimeterVerts.slice(1).forEach(v=>{{ const [px,py]=toDisplay(...v); ctx.lineTo(px,py); }});
    ctx.stroke(); ctx.setLineDash([]);
    perimeterVerts.forEach((v,i)=>{{
      const [vx,vy]=toDisplay(...v);
      ctx.fillStyle=i===0?"#ffd700":"#fff";
      ctx.beginPath(); ctx.arc(vx,vy,HANDLE_RADIUS,0,2*Math.PI); ctx.fill();
    }});
  }}

  if ((mode==="rectangle"||mode==="oval")&&dragStart&&dragCurrent) {{
    const verts = mode==="rectangle"
      ? [[dragStart[0],dragStart[1]],[dragCurrent[0],dragStart[1]],
         [dragCurrent[0],dragCurrent[1]],[dragStart[0],dragCurrent[1]]]
      : ovalVertices(...dragStart,...dragCurrent,32);
    ctx.strokeStyle=maskType==="include"?INCLUDE_STROKE:IGNORE_STROKE;
    ctx.fillStyle  =maskType==="include"?INCLUDE_FILL  :IGNORE_FILL;
    ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.beginPath();
    const [d0,e0]=toDisplay(...verts[0]); ctx.moveTo(d0,e0);
    verts.slice(1).forEach(v=>{{ const [px,py]=toDisplay(...v); ctx.lineTo(px,py); }});
    ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
  }}
}}

function drawShape(shape) {{
  const isSel  = shape.id===selectedId;
  const fill   = shape.proposed?PROPOSED_FILL  :shape.type==="include"?INCLUDE_FILL  :IGNORE_FILL;
  const stroke = shape.proposed?PROPOSED_STROKE:shape.type==="include"?INCLUDE_STROKE:IGNORE_STROKE;

  ctx.beginPath();
  const [fx,fy]=toDisplay(...shape.vertices[0]); ctx.moveTo(fx,fy);
  shape.vertices.slice(1).forEach(v=>{{ const [px,py]=toDisplay(...v); ctx.lineTo(px,py); }});
  ctx.closePath();
  ctx.fillStyle=fill; ctx.strokeStyle=stroke; ctx.lineWidth=isSel?2:1.5;
  ctx.fill(); ctx.stroke();

  if (isSel) {{
    // ---- bounding box ----
    const bb = shapeBoundingBox(shape.vertices);
    const [blx,bly]=toDisplay(bb.x,bb.y);
    const bbw=bb.w*displayScale, bbh=bb.h*displayScale;
    ctx.strokeStyle=BB_STROKE; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.strokeRect(blx,bly,bbw,bbh); ctx.setLineDash([]);

    // bb handles
    bbHandles(bb).forEach((h,i)=>{{
      const [hx,hy]=toDisplay(...h);
      ctx.fillStyle="#c0d8ff"; ctx.strokeStyle="#fff"; ctx.lineWidth=1;
      ctx.beginPath(); ctx.arc(hx,hy,BB_HANDLE_RADIUS,0,2*Math.PI);
      ctx.fill(); ctx.stroke();
    }});

    // ---- vertex handles ----
    shape.vertices.forEach(v=>{{
      const [vx,vy]=toDisplay(...v);
      ctx.fillStyle="#fff"; ctx.strokeStyle=stroke; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(vx,vy,HANDLE_RADIUS,0,2*Math.PI);
      ctx.fill(); ctx.stroke();
    }});

    // ---- rotation handle ----
    const [rhx,rhy]=rotHandlePos(shape);
    const [cx]=centroid(shape.vertices);
    const minY=Math.min(...shape.vertices.map(v=>v[1]));
    const [cdx,cdy]=toDisplay(cx,minY);
    ctx.strokeStyle="#ffd700"; ctx.lineWidth=1; ctx.setLineDash([3,2]);
    ctx.beginPath(); ctx.moveTo(cdx,cdy); ctx.lineTo(rhx,rhy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="#ffd700"; ctx.strokeStyle="#fff"; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(rhx,rhy,HANDLE_RADIUS,0,2*Math.PI);
    ctx.fill(); ctx.stroke();
  }}
}}

// ============================================================
// Close / shutdown
// ============================================================
function requestClose() {{
  const msg    = document.getElementById("shutdown-msg");
  const saveBtn = document.getElementById("btn-save-close");
  if (hasUnsaved()) {{
    msg.textContent="You have unsaved shapes. Save before closing?";
    saveBtn.style.display="";
  }} else {{
    msg.textContent="Close the mask editor?";
    saveBtn.style.display="none";
  }}
  document.getElementById("shutdown-overlay").classList.add("visible");
}}
function cancelClose() {{ document.getElementById("shutdown-overlay").classList.remove("visible"); }}
async function saveAndClose() {{ await saveAll(); closeNow(); }}
async function closeNow() {{
  await fetch("/shutdown",{{method:"POST"}});
  document.body.innerHTML=
    '<div style="display:flex;align-items:center;justify-content:center;height:100vh;' +
    'background:#1a1a2e;color:#a0c4ff;font-family:system-ui;font-size:18px">' +
    'Editor closed — you can close this tab.</div>';
}}

// ============================================================
// Auto-detect
// ============================================================
async function runDetect() {{
  setStatus("Running detection…");
  const hsvColor=hexToHsv(document.getElementById("det-color").value);
  const body={{
    diameter_cm:    parseFloat(document.getElementById("det-diameter").value),
    thickness_cm:   parseFloat(document.getElementById("det-thickness").value),
    expected_count: parseInt(document.getElementById("det-count").value),
    hue_center:     hsvColor.h,
    hue_tolerance:  parseInt(document.getElementById("det-hue-tol").value),
    saturation_min: 80,
    value_max:      parseInt(document.getElementById("det-brightness").value),
  }};
  const res=await fetch("/masks-frame/masks/detect",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(body)}});
  const d=await res.json();
  d.polygons.forEach(verts=>addShape(verts,maskType,null,true));
  setStatus(`Detection complete: ${{d.count}} circle(s) found.`);
}}

async function runDetectDebug() {{
  setStatus("Running detection debug…");
  const hsvColor=hexToHsv(document.getElementById("det-color").value);
  const body={{
    diameter_cm:    parseFloat(document.getElementById("det-diameter").value),
    thickness_cm:   parseFloat(document.getElementById("det-thickness").value),
    expected_count: parseInt(document.getElementById("det-count").value),
    hue_center:     hsvColor.h,
    hue_tolerance:  parseInt(document.getElementById("det-hue-tol").value),
    saturation_min: 80,
    value_max:      parseInt(document.getElementById("det-brightness").value),
  }};
  const res=await fetch("/masks-frame/masks/detect-debug",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(body)}});
  const d=await res.json();

  document.getElementById("debug-area-range").textContent =
    `${{d.area_min.toFixed(0)}}–${{d.area_max.toFixed(0)}} px² (ideal: ${{d.ideal_area.toFixed(0)}} px²)`;

  const grid = document.getElementById("debug-grid");
  grid.innerHTML = "";
  const stages = [
    ["1. Colour mask (HSV threshold)", d.colour_mask_image],
    ["2. Cleaned mask (after morphology)", d.cleaned_mask_image],
    ["3. Contours found (green=accepted, red=rejected)", d.contours_image],
  ];
  stages.forEach(([label, imageB64]) => {{
    const cell = document.createElement("div");
    cell.className = "debug-cell";
    const labelDiv = document.createElement("div");
    labelDiv.className = "debug-cell-label";
    labelDiv.textContent = label;
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + imageB64;
    cell.appendChild(labelDiv);
    cell.appendChild(img);
    grid.appendChild(cell);
  }});

  const rows = document.getElementById("debug-contour-rows");
  rows.innerHTML = "";
  if (d.contour_details.length === 0) {{
    rows.innerHTML = '<tr><td colspan="4" style="color:#dc5050;">No contours found at all — ' +
      'the colour mask is likely empty. Try widening hue tolerance or raising max brightness.</td></tr>';
  }} else {{
    d.contour_details.forEach((c, i) => {{
      const row = document.createElement("tr");
      const resultClass = c.accepted ? "debug-accepted" : "debug-rejected";
      row.innerHTML = `<td>${{i+1}}</td><td>${{c.area.toFixed(0)}}</td>` +
        `<td>${{c.circularity.toFixed(2)}}</td><td class="${{resultClass}}">${{c.reason}}</td>`;
      rows.appendChild(row);
    }});
  }}

  document.getElementById("debug-overlay").classList.add("visible");
  setStatus(`Debug complete: ${{d.contour_details.length}} contour(s) examined.`);
}}

function closeDebugOverlay() {{
  document.getElementById("debug-overlay").classList.remove("visible");
}}

async function saveDetectParams() {{
  const hsvColor=hexToHsv(document.getElementById("det-color").value);
  const params={{
    diameter_cm:    parseFloat(document.getElementById("det-diameter").value),
    thickness_cm:   parseFloat(document.getElementById("det-thickness").value),
    expected_count: parseInt(document.getElementById("det-count").value),
    hue_center:     hsvColor.h,
    hue_tolerance:  parseInt(document.getElementById("det-hue-tol").value),
    value_max:      parseInt(document.getElementById("det-brightness").value),
  }};
  await fetch("/masks-frame/project/auto-detect-params",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{params}})}});
  setStatus("Auto-detect params saved to project.yaml.");
}}

// ============================================================
// Batch defaults
// ============================================================
let currentBatchName = null;

async function checkBatch() {{
  const res = await fetch("/masks-frame/masks/batch"), d = await res.json();
  currentBatchName = d.batch_name;
  const label = document.getElementById("batch-name-label");
  label.textContent = currentBatchName
    ? `Batch: ${{currentBatchName}}`
    : "This video is not in any batch.";
}}

async function loadBatchDefaults() {{
  const res = await fetch("/masks-frame/masks/batch"), d = await res.json();
  if (!d.batch_name) {{ setStatus("This video is not in any batch."); return; }}
  d.shapes.forEach(s => addShape(s.vertices, s.type, null, false));
  setStatus(`Loaded ${{d.shapes.length}} shape(s) from batch '${{d.batch_name}}'.`);
}}

async function saveAsBatchDefaults() {{
  if (!currentBatchName) {{ setStatus("This video is not in any batch — cannot save batch defaults."); return; }}
  const body = {{shapes: shapes.map(s=>{{return{{type:s.type,vertices:s.vertices}}}})}};
  await fetch("/masks-frame/masks/save-batch-defaults", {{method:"POST",
    headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(body)}});
  setStatus(`Saved as defaults for batch '${{currentBatchName}}'.`);
}}

// ============================================================
// Defaults
// ============================================================
async function loadDefaults() {{
  const res=await fetch("/masks-frame/masks/defaults"),d=await res.json();
  d.shapes.forEach(s=>addShape(s.vertices,s.type,null,false));
  setStatus(`Loaded ${{d.shapes.length}} default shape(s).`);
}}

async function saveAsDefaults() {{
  const body={{shapes:shapes.map(s=>{{return{{type:s.type,vertices:s.vertices}}}})}};
  await fetch("/masks-frame/masks/save-defaults",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(body)}});
  setStatus("Saved as project defaults.");
}}

// ============================================================
// Save all
// ============================================================
async function saveAll() {{
  let saved=0;
  for (const shape of shapes) {{
    if (!shape.filename) {{
      const res=await fetch("/masks-frame/masks/save",{{method:"POST",headers:{{"Content-Type":"application/json"}},
        body:JSON.stringify({{vertices:shape.vertices,mask_type:shape.type,prefix:VIDEO_NAME}})}});
      const d=await res.json();
      shape.filename=d.filename; shape.proposed=false; saved++;
    }}
  }}
  refreshShapeList();
  setStatus(`Saved ${{saved}} shape(s). Total: ${{shapes.length}}.`);
}}

// ============================================================
// Utilities
// ============================================================
function setStatus(msg) {{ document.getElementById("status").textContent=msg; }}

function hexToHsv(hex) {{
  const r=parseInt(hex.slice(1,3),16)/255,g=parseInt(hex.slice(3,5),16)/255,b=parseInt(hex.slice(5,7),16)/255;
  const max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min;
  let h=0;
  if(d!==0){{if(max===r)h=((g-b)/d)%6;else if(max===g)h=(b-r)/d+2;else h=(r-g)/d+4;}}
  h=Math.round(h*30)%180;
  return {{h:h<0?h+180:h}};
}}
</script>
</body>
</html>"""

# =============================================================================
# Helpers
# =============================================================================

def _read_polygon_csv_raw(polygon_file: Path) -> list:
    vertices = []
    for line in polygon_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) == 2:
            try:
                vertices.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    return vertices
