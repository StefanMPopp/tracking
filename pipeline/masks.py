"""
masks.py — Interactive mask editor for TRex tracking regions.

Usage:
    uv run python pipeline/masks.py --project /path/to/project --video pain_test
    uv run python pipeline/masks.py --project /path/to/project --video pain_test --frame 500

Opens a browser-based editor to create, edit, and save include/ignore mask
polygons for a specific video. Masks are saved to projects/{name}/masks/.

Shapes:
  - Click perimeter: place vertices one by one
  - Rectangle / Oval: click and drag
  - Auto-detect: CV-based detection of coloured circles

All shapes can be moved, reshaped (drag vertices), and rotated.
Right-click a vertex to delete it. Right-click a shape body to delete the shape.

Project defaults are stored as default_*.csv and can be loaded into any video
session as a starting point.
"""

import argparse
import base64
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from _masks import (
    delete_polygon,
    detect_circles,
    load_default_masks,
    load_masks_for_video,
    save_default_masks,
    save_polygon,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# App state (module-level, set at startup)
# =============================================================================

app = FastAPI()

APP_STATE: dict = {}   # populated in main() before server starts


# =============================================================================
# Pydantic models
# =============================================================================

class PolygonSaveRequest(BaseModel):
    vertices: list[list[float]]
    mask_type: str        # 'include' or 'ignore'
    prefix: str           # video_name or 'default'


class DefaultSaveRequest(BaseModel):
    shapes: list[dict]    # [{"type": "include"|"ignore", "vertices": [[x,y],...]}]


class DetectRequest(BaseModel):
    diameter_cm: float
    thickness_cm: float
    hue_center: int
    hue_tolerance: int
    saturation_min: int = 80
    value_max: int = 120
    expected_count: int = 1


class AutoDetectParamsRequest(BaseModel):
    params: dict


# =============================================================================
# API routes
# =============================================================================

@app.get("/frame")
def get_frame():
    """Return the display frame as a base64-encoded JPEG."""
    frame      = APP_STATE["frame"]
    _, buffer  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64        = base64.b64encode(buffer).decode("utf-8")
    return JSONResponse({
        "image":        b64,
        "frame_width":  frame.shape[1],
        "frame_height": frame.shape[0],
    })


@app.get("/masks")
def get_masks():
    """Return existing per-video mask polygons and their filenames."""
    masks_dir  = APP_STATE["masks_dir"]
    video_name = APP_STATE["video_name"]

    include_files = sorted(masks_dir.glob(f"{video_name}_include_*.csv"))
    ignore_files  = sorted(masks_dir.glob(f"{video_name}_ignore_*.csv"))

    shapes = []
    for mask_file in include_files + ignore_files:
        mask_type = "include" if "_include_" in mask_file.name else "ignore"
        vertices  = _read_polygon_csv_raw(mask_file)
        shapes.append({
            "filename": mask_file.name,
            "type":     mask_type,
            "vertices": vertices,
        })
    return JSONResponse({"shapes": shapes})


@app.get("/masks/defaults")
def get_default_masks():
    """Return project-default mask polygons and their filenames."""
    masks_dir = APP_STATE["masks_dir"]
    shapes    = []
    for mask_file in sorted(masks_dir.glob("default_*.csv")):
        mask_type = "include" if "_include_" in mask_file.name else "ignore"
        vertices  = _read_polygon_csv_raw(mask_file)
        shapes.append({
            "filename": mask_file.name,
            "type":     mask_type,
            "vertices": vertices,
        })
    return JSONResponse({"shapes": shapes})


@app.post("/masks/save")
def save_mask(request: PolygonSaveRequest):
    """Save one polygon to a new numbered CSV file."""
    masks_dir = APP_STATE["masks_dir"]
    if request.mask_type not in ("include", "ignore"):
        raise HTTPException(status_code=400, detail="mask_type must be 'include' or 'ignore'")
    filename = save_polygon(
        polygon=request.vertices,
        prefix=request.prefix,
        mask_type=request.mask_type,
        masks_dir=masks_dir,
    )
    return JSONResponse({"filename": filename})


@app.delete("/masks/{filename}")
def delete_mask(filename: str):
    """Delete a mask polygon file."""
    masks_dir = APP_STATE["masks_dir"]
    delete_polygon(filename, masks_dir)
    return JSONResponse({"deleted": filename})


@app.post("/masks/save-defaults")
def save_defaults(request: DefaultSaveRequest):
    """Overwrite project default mask files with the given shapes."""
    masks_dir = APP_STATE["masks_dir"]
    save_default_masks(request.shapes, masks_dir)
    return JSONResponse({"saved": len(request.shapes)})


@app.post("/masks/detect")
def run_detection(request: DetectRequest):
    """Run CV circle detection on the current frame."""
    frame            = APP_STATE["frame"]
    meta_real_width  = APP_STATE["meta_real_width"]
    frame_width      = frame.shape[1]

    polygons = detect_circles(
        frame=frame,
        diameter_cm=request.diameter_cm,
        thickness_cm=request.thickness_cm,
        meta_real_width=meta_real_width,
        expected_count=request.expected_count,
        hue_center=request.hue_center,
        hue_tolerance=request.hue_tolerance,
        saturation_min=request.saturation_min,
        value_max=request.value_max,
    )
    return JSONResponse({"polygons": polygons, "count": len(polygons)})


@app.post("/project/auto-detect-params")
def save_auto_detect_params(request: AutoDetectParamsRequest):
    """Persist auto-detect parameters back to project.yaml."""
    project_yaml_file = APP_STATE["project_dir"] / "project.yaml"
    project_config    = yaml.safe_load(project_yaml_file.read_text())
    project_config["auto_detect_circles"] = request.params
    project_yaml_file.write_text(
        yaml.dump(project_config, default_flow_style=False, sort_keys=False)
    )
    logger.info("Saved auto-detect params to project.yaml: %s", request.params)
    return JSONResponse({"saved": True})


@app.get("/project/auto-detect-params")
def get_auto_detect_params():
    """Return the auto-detect params currently stored in project.yaml."""
    project_yaml_file = APP_STATE["project_dir"] / "project.yaml"
    project_config    = yaml.safe_load(project_yaml_file.read_text())
    params            = project_config.get("auto_detect_circles") or {}
    return JSONResponse(params)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_build_html(
        video_name=APP_STATE["video_name"],
        frame_width=APP_STATE["frame"].shape[1],
        frame_height=APP_STATE["frame"].shape[0],
    ))


# =============================================================================
# HTML + JS frontend
# =============================================================================

def _build_html(video_name: str, frame_width: int, frame_height: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mask Editor — {video_name}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0;
       display: flex; height: 100vh; overflow: hidden; }}

/* ---- sidebar ---- */
#sidebar {{
  width: 260px; min-width: 260px; background: #16213e;
  display: flex; flex-direction: column; padding: 12px; gap: 10px;
  overflow-y: auto; border-right: 1px solid #0f3460;
}}
h2 {{ font-size: 13px; color: #a0c4ff; text-transform: uppercase;
     letter-spacing: 1px; margin-bottom: 2px; }}
.section {{ background: #0f3460; border-radius: 6px; padding: 10px; }}
.section h3 {{ font-size: 11px; color: #7ec8e3; text-transform: uppercase;
              letter-spacing: 0.8px; margin-bottom: 8px; }}

button {{
  width: 100%; padding: 7px 10px; border: none; border-radius: 5px;
  cursor: pointer; font-size: 12px; font-weight: 600; transition: opacity .15s;
}}
button:hover {{ opacity: 0.85; }}
button.active {{ outline: 2px solid #fff; }}

.btn-include {{ background: #2d6a4f; color: #d8f3dc; }}
.btn-ignore  {{ background: #9d0208; color: #ffccd5; }}
.btn-neutral {{ background: #415a77; color: #e0e0e0; }}
.btn-detect  {{ background: #e76f51; color: #fff; }}
.btn-default {{ background: #5c4033; color: #ffe0b2; }}
.btn-save-def {{ background: #6a0572; color: #f3e5f5; }}

.mode-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }}

label {{ font-size: 11px; color: #aaa; display: block; margin-bottom: 2px; }}
input[type=number], input[type=range] {{
  width: 100%; padding: 4px 6px; background: #1a1a2e; border: 1px solid #415a77;
  border-radius: 4px; color: #e0e0e0; font-size: 12px;
}}
input[type=color] {{
  width: 100%; height: 30px; padding: 2px; background: #1a1a2e;
  border: 1px solid #415a77; border-radius: 4px; cursor: pointer;
}}
.field {{ margin-bottom: 6px; }}

#status {{
  font-size: 11px; color: #7ec8e3; padding: 6px;
  background: #0f3460; border-radius: 4px; min-height: 32px;
}}
#shape-list {{ max-height: 200px; overflow-y: auto; }}
.shape-item {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 6px; border-radius: 4px; margin-bottom: 3px;
  font-size: 11px; cursor: pointer;
}}
.shape-item:hover {{ background: #1a3a5c; }}
.shape-item.selected {{ background: #1a3a5c; outline: 1px solid #a0c4ff; }}
.shape-item .dot {{
  width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-right: 6px;
}}
.shape-item .del-btn {{
  background: none; border: none; color: #ff6b6b; cursor: pointer;
  font-size: 13px; width: auto; padding: 0 4px;
}}

/* ---- canvas area ---- */
#canvas-wrap {{
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center; overflow: hidden;
}}
canvas {{ cursor: crosshair; display: block; max-width: 100%; max-height: 100%; }}
#canvas-wrap.mode-select canvas {{ cursor: default; }}

/* ---- type toggle ---- */
#type-bar {{
  display: flex; gap: 6px; width: 100%;
}}
#type-bar button {{ flex: 1; }}
</style>
</head>
<body>

<div id="sidebar">
  <h2>Mask Editor</h2>
  <div id="status">Loading…</div>

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
    <button class="btn-detect" onclick="runDetect()">🔍 Detect</button>
    <button class="btn-neutral" style="margin-top:5px" onclick="saveDetectParams()">💾 Save params to project</button>
  </div>

  <div class="section">
    <h3>Project defaults</h3>
    <button class="btn-default" onclick="loadDefaults()">📂 Load project defaults</button>
    <button class="btn-save-def" style="margin-top:5px" onclick="saveAsDefaults()">⭐ Save current as defaults</button>
  </div>

  <div class="section">
    <h3>Shapes (<span id="shape-count">0</span>)</h3>
    <div id="shape-list"></div>
  </div>

  <button class="btn-neutral" onclick="saveAll()" style="background:#1b4332;color:#d8f3dc;margin-top:auto">
    💾 Save all unsaved shapes
  </button>
</div>

<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>

<script>
// ============================================================
// State
// ============================================================
const HANDLE_RADIUS    = 6;
const ROT_HANDLE_DIST  = 24;
const INCLUDE_FILL     = "rgba(45,180,100,0.25)";
const INCLUDE_STROKE   = "rgba(45,220,80,0.9)";
const IGNORE_FILL      = "rgba(220,30,30,0.25)";
const IGNORE_STROKE    = "rgba(220,50,50,0.9)";
const PROPOSED_FILL    = "rgba(255,200,0,0.18)";
const PROPOSED_STROKE  = "rgba(255,200,0,0.85)";

let mode       = "perimeter";   // perimeter | rectangle | oval | select
let maskType   = "include";
let shapes     = [];            // {{id, type, vertices, filename, proposed}}
let selectedId = null;
let nextId     = 1;

// Perimeter drawing state
let perimeterVerts = [];

// Drag-rect / oval state
let dragStart = null;
let dragCurrent = null;

// Select / drag state
let dragging        = false;
let dragTarget      = null;   // {{shapeId, vertexIdx|-1 (whole shape)|-2 (rotation)}}
let dragOffset      = null;
let shapeStartVerts = null;
let rotCenterAtDrag = null;
let rotStartAngle   = null;

// Canvas / image
let canvas, ctx, img;
let displayScale = 1;
let nativeW, nativeH;

// ============================================================
// Init
// ============================================================
window.onload = async () => {{
  canvas = document.getElementById("canvas");
  ctx    = canvas.getContext("2d");
  setMode("perimeter");
  setMaskType("include");
  wireSliders();
  await loadFrame();
  await loadExistingMasks();
  await loadAutoDetectParams();
  setStatus("Ready. Video: {video_name}");
}};

function wireSliders() {{
  const pairs = [["det-hue-tol","det-hue-tol-val"],["det-brightness","det-brightness-val"]];
  pairs.forEach(([sid, vid]) => {{
    const el = document.getElementById(sid);
    const vl = document.getElementById(vid);
    el.oninput = () => vl.textContent = el.value;
  }});
}}

async function loadFrame() {{
  const res  = await fetch("/frame");
  const data = await res.json();
  nativeW    = data.frame_width;
  nativeH    = data.frame_height;
  img        = new Image();
  img.src    = "data:image/jpeg;base64," + data.image;
  await new Promise(r => img.onload = r);
  resizeCanvas();
  window.onresize = resizeCanvas;
}}

function resizeCanvas() {{
  const wrap  = document.getElementById("canvas-wrap");
  const maxW  = wrap.clientWidth  - 10;
  const maxH  = wrap.clientHeight - 10;
  displayScale = Math.min(maxW / nativeW, maxH / nativeH, 1);
  canvas.width  = Math.round(nativeW * displayScale);
  canvas.height = Math.round(nativeH * displayScale);
  redraw();
}}

async function loadExistingMasks() {{
  const res  = await fetch("/masks");
  const data = await res.json();
  data.shapes.forEach(s => addShape(s.vertices, s.type, s.filename));
}}

async function loadAutoDetectParams() {{
  const res  = await fetch("/project/auto-detect-params");
  const data = await res.json();
  if (data.diameter_cm)  document.getElementById("det-diameter").value  = data.diameter_cm;
  if (data.thickness_cm) document.getElementById("det-thickness").value = data.thickness_cm;
  if (data.expected_count) document.getElementById("det-count").value   = data.expected_count;
}}

// ============================================================
// Shape management
// ============================================================
function addShape(vertices, type, filename=null, proposed=false) {{
  const id = nextId++;
  shapes.push({{ id, type, vertices: vertices.map(v=>[...v]), filename, proposed }});
  selectedId = id;
  refreshShapeList();
  redraw();
  return id;
}}

function removeShape(id) {{
  const shape = shapes.find(s => s.id === id);
  if (!shape) return;
  if (shape.filename) {{
    fetch(`/masks/${{shape.filename}}`, {{method:"DELETE"}});
  }}
  shapes = shapes.filter(s => s.id !== id);
  if (selectedId === id) selectedId = null;
  refreshShapeList();
  redraw();
}}

function refreshShapeList() {{
  const list = document.getElementById("shape-list");
  document.getElementById("shape-count").textContent = shapes.length;
  list.innerHTML = "";
  shapes.forEach(s => {{
    const div  = document.createElement("div");
    div.className = "shape-item" + (s.id === selectedId ? " selected" : "");
    div.onclick   = () => {{ selectedId = s.id; refreshShapeList(); redraw(); }};
    const dot  = document.createElement("span");
    dot.className = "dot";
    dot.style.background = s.proposed ? "#ffd700"
                         : s.type === "include" ? "#2db864" : "#dc1e1e";
    const label = document.createElement("span");
    label.style.flex = "1";
    label.textContent = (s.filename || (s.proposed ? "proposed" : "unsaved"))
                      + " (" + s.vertices.length + "v)";
    const del  = document.createElement("button");
    del.className   = "del-btn";
    del.textContent = "✕";
    del.onclick = e => {{ e.stopPropagation(); removeShape(s.id); }};
    div.appendChild(dot);
    div.appendChild(label);
    div.appendChild(del);
    list.appendChild(div);
  }});
}}

// ============================================================
// Mode / type
// ============================================================
function setMode(m) {{
  mode = m;
  perimeterVerts = [];
  dragStart = dragCurrent = null;
  ["perimeter","rectangle","oval","select"].forEach(n => {{
    document.getElementById("btn-"+n).classList.toggle("active", n===m);
  }});
  document.getElementById("canvas-wrap").className =
    m === "select" ? "mode-select" : "";
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
// Canvas events
// ============================================================
canvas.addEventListener("mousedown", onMouseDown);
canvas.addEventListener("mousemove", onMouseMove);
canvas.addEventListener("mouseup",   onMouseUp);
canvas.addEventListener("dblclick",  onDblClick);
canvas.addEventListener("contextmenu", onRightClick);

function onMouseDown(e) {{
  e.preventDefault();
  if (e.button !== 0) return;
  const [dx, dy] = canvasXY(e);
  const [nx, ny] = toNative(dx, dy);

  if (mode === "select") {{
    const hit = hitTest(dx, dy);
    if (hit) {{
      dragging        = true;
      dragTarget      = hit;
      dragOffset      = [dx, dy];
      selectedId      = hit.shapeId;
      const shape     = shapes.find(s => s.id === hit.shapeId);
      shapeStartVerts = shape.vertices.map(v=>[...v]);
      if (hit.vertexIdx === -2) {{
        rotCenterAtDrag = centroid(shape.vertices);
        rotStartAngle   = Math.atan2(dy - rotCenterAtDrag[1]*displayScale,
                                      dx - rotCenterAtDrag[0]*displayScale);
      }}
      refreshShapeList();
      redraw();
    }} else {{
      selectedId = null;
      refreshShapeList();
      redraw();
    }}
    return;
  }}

  if (mode === "perimeter") {{
    // Snap to first vertex if close enough
    if (perimeterVerts.length >= 3) {{
      const [fx, fy] = toDisplay(...perimeterVerts[0]);
      if (Math.hypot(dx-fx, dy-fy) < HANDLE_RADIUS*2) {{
        closePerimeter();
        return;
      }}
    }}
    perimeterVerts.push([nx, ny]);
    redraw();
    return;
  }}

  if (mode === "rectangle" || mode === "oval") {{
    dragStart   = [nx, ny];
    dragCurrent = [nx, ny];
    return;
  }}
}}

function onMouseMove(e) {{
  const [dx, dy] = canvasXY(e);
  const [nx, ny] = toNative(dx, dy);

  if (mode === "rectangle" || mode === "oval") {{
    if (dragStart) {{ dragCurrent = [nx, ny]; redraw(); }}
    return;
  }}

  if (mode === "select" && dragging && dragTarget) {{
    const shape = shapes.find(s => s.id === dragTarget.shapeId);
    if (!shape) return;

    const ddx = (dx - dragOffset[0]) / displayScale;
    const ddy = (dy - dragOffset[1]) / displayScale;

    if (dragTarget.vertexIdx === -2) {{
      // Rotation
      const [cx, cy] = rotCenterAtDrag;
      const currentAngle = Math.atan2(dy - cy*displayScale, dx - cx*displayScale);
      const delta        = currentAngle - rotStartAngle;
      shape.vertices = shapeStartVerts.map(([vx,vy]) => {{
        const rx = vx - cx, ry = vy - cy;
        return [cx + rx*Math.cos(delta) - ry*Math.sin(delta),
                cy + rx*Math.sin(delta) + ry*Math.cos(delta)];
      }});
    }} else if (dragTarget.vertexIdx === -1) {{
      // Move whole shape
      shape.vertices = shapeStartVerts.map(([vx,vy]) => [vx+ddx, vy+ddy]);
    }} else {{
      // Move single vertex
      const sv = shapeStartVerts[dragTarget.vertexIdx];
      shape.vertices[dragTarget.vertexIdx] = [sv[0]+ddx, sv[1]+ddy];
    }}
    redraw();
    return;
  }}

  if (mode === "perimeter" && perimeterVerts.length > 0) {{
    redraw();
    // Draw rubber-band line to cursor
    const [lx, ly] = toDisplay(...perimeterVerts[perimeterVerts.length-1]);
    ctx.strokeStyle = "#ffd700";
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4,4]);
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(dx, dy);
    ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function onMouseUp(e) {{
  if (mode === "rectangle" || mode === "oval") {{
    if (!dragStart || !dragCurrent) return;
    const [x0,y0] = dragStart, [x1,y1] = dragCurrent;
    if (Math.abs(x1-x0) < 3 && Math.abs(y1-y0) < 3) {{
      dragStart = dragCurrent = null; return;
    }}
    const verts = mode === "rectangle"
      ? [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
      : ovalVertices(x0,y0,x1,y1,32);
    addShape(verts, maskType);
    dragStart = dragCurrent = null;
    return;
  }}
  if (mode === "select" && dragging) {{
    dragging = false;
    dragTarget = null;
  }}
}}

function onDblClick(e) {{
  if (mode === "perimeter" && perimeterVerts.length >= 3) {{
    closePerimeter();
  }}
}}

function onRightClick(e) {{
  e.preventDefault();
  const [dx, dy] = canvasXY(e);
  const [nx, ny] = toNative(dx, dy);

  if (mode === "select") {{
    const hit = hitTest(dx, dy);
    if (!hit) return;
    if (hit.vertexIdx >= 0) {{
      // Delete that vertex
      const shape = shapes.find(s => s.id === hit.shapeId);
      if (shape && shape.vertices.length > 3) {{
        shape.vertices.splice(hit.vertexIdx, 1);
        redraw();
      }}
    }} else {{
      removeShape(hit.shapeId);
    }}
    return;
  }}

  if (mode === "perimeter" && perimeterVerts.length > 0) {{
    perimeterVerts.pop();
    redraw();
  }}
}}

// ============================================================
// Perimeter helpers
// ============================================================
function closePerimeter() {{
  if (perimeterVerts.length >= 3) {{
    addShape([...perimeterVerts], maskType);
  }}
  perimeterVerts = [];
  redraw();
}}

document.addEventListener("keydown", e => {{
  if (e.key === "Enter" && mode === "perimeter" && perimeterVerts.length >= 3) {{
    closePerimeter();
  }}
  if (e.key === "Escape") {{
    perimeterVerts = [];
    dragStart = dragCurrent = null;
    redraw();
  }}
  // Hotkeys
  if (e.key === "s" || e.key === "S") setMode("select");
  if (e.key === "p" || e.key === "P") setMode("perimeter");
  if (e.key === "r" || e.key === "R") setMode("rectangle");
  if (e.key === "o" || e.key === "O") setMode("oval");
  if (e.key === "i" || e.key === "I") setMaskType("include");
  if (e.key === "g" || e.key === "G") setMaskType("ignore");
  if ((e.key === "Delete" || e.key === "Backspace") && selectedId) {{
    removeShape(selectedId);
  }}
}});

// ============================================================
// Hit testing
// ============================================================
function hitTest(dx, dy) {{
  // Priority: rotation handle > vertex > shape interior
  for (const shape of [...shapes].reverse()) {{
    // Rotation handle
    const [rcx, rcy] = centroid(shape.vertices);
    const [rdx, rdy] = toDisplay(rcx, rcy);
    const [rhx, rhy] = rotHandlePos(shape);
    if (Math.hypot(dx-rhx, dy-rhy) < HANDLE_RADIUS+2)
      return {{shapeId: shape.id, vertexIdx: -2}};
    // Vertices
    for (let i = 0; i < shape.vertices.length; i++) {{
      const [vdx, vdy] = toDisplay(...shape.vertices[i]);
      if (Math.hypot(dx-vdx, dy-vdy) < HANDLE_RADIUS+2)
        return {{shapeId: shape.id, vertexIdx: i}};
    }}
    // Interior
    if (pointInPolygon([dx/displayScale, dy/displayScale], shape.vertices))
      return {{shapeId: shape.id, vertexIdx: -1}};
  }}
  return null;
}}

function pointInPolygon([px,py], verts) {{
  let inside = false;
  for (let i=0, j=verts.length-1; i<verts.length; j=i++) {{
    const [xi,yi]=verts[i], [xj,yj]=verts[j];
    if (((yi>py)!=(yj>py)) && (px < (xj-xi)*(py-yi)/(yj-yi)+xi)) inside=!inside;
  }}
  return inside;
}}

// ============================================================
// Geometry helpers
// ============================================================
function centroid(verts) {{
  const cx = verts.reduce((s,[x])=>s+x,0)/verts.length;
  const cy = verts.reduce((s,[,y])=>s+y,0)/verts.length;
  return [cx, cy];
}}

function rotHandlePos(shape) {{
  const [cx,cy] = centroid(shape.vertices);
  // Find topmost vertex for the rotation handle position
  const minY = Math.min(...shape.vertices.map(v=>v[1]));
  const [dcx, dcy] = toDisplay(cx, minY);
  return [dcx, dcy - ROT_HANDLE_DIST];
}}

function ovalVertices(x0,y0,x1,y1,n) {{
  const cx=(x0+x1)/2, cy=(y0+y1)/2, rx=Math.abs(x1-x0)/2, ry=Math.abs(y1-y0)/2;
  return Array.from({{length:n}}, (_,i) => [
    cx + rx*Math.cos(2*Math.PI*i/n),
    cy + ry*Math.sin(2*Math.PI*i/n),
  ]);
}}

// ============================================================
// Drawing
// ============================================================
function redraw() {{
  if (!img) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  shapes.forEach(shape => drawShape(shape));

  // In-progress perimeter
  if (mode === "perimeter" && perimeterVerts.length > 0) {{
    ctx.strokeStyle = maskType === "include" ? INCLUDE_STROKE : IGNORE_STROKE;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4,4]);
    ctx.beginPath();
    const [fx,fy] = toDisplay(...perimeterVerts[0]);
    ctx.moveTo(fx,fy);
    perimeterVerts.slice(1).forEach(v => {{ const [px,py]=toDisplay(...v); ctx.lineTo(px,py); }});
    ctx.stroke();
    ctx.setLineDash([]);
    perimeterVerts.forEach((v,i) => {{
      const [vx,vy] = toDisplay(...v);
      ctx.fillStyle = i===0 ? "#ffd700" : "#fff";
      ctx.beginPath(); ctx.arc(vx,vy,HANDLE_RADIUS,0,2*Math.PI); ctx.fill();
    }});
  }}

  // In-progress rect/oval
  if ((mode==="rectangle"||mode==="oval") && dragStart && dragCurrent) {{
    const verts = mode==="rectangle"
      ? [[dragStart[0],dragStart[1]],[dragCurrent[0],dragStart[1]],
         [dragCurrent[0],dragCurrent[1]],[dragStart[0],dragCurrent[1]]]
      : ovalVertices(...dragStart,...dragCurrent,32);
    ctx.strokeStyle = maskType==="include" ? INCLUDE_STROKE : IGNORE_STROKE;
    ctx.fillStyle   = maskType==="include" ? INCLUDE_FILL   : IGNORE_FILL;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([5,3]);
    ctx.beginPath();
    const [dx0,dy0] = toDisplay(...verts[0]); ctx.moveTo(dx0,dy0);
    verts.slice(1).forEach(v=>{{ const [px,py]=toDisplay(...v); ctx.lineTo(px,py); }});
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function drawShape(shape) {{
  const isSelected = shape.id === selectedId;
  const fill   = shape.proposed ? PROPOSED_FILL
               : shape.type==="include" ? INCLUDE_FILL  : IGNORE_FILL;
  const stroke = shape.proposed ? PROPOSED_STROKE
               : shape.type==="include" ? INCLUDE_STROKE : IGNORE_STROKE;

  ctx.beginPath();
  const [fx,fy] = toDisplay(...shape.vertices[0]); ctx.moveTo(fx,fy);
  shape.vertices.slice(1).forEach(v=>{{ const [px,py]=toDisplay(...v); ctx.lineTo(px,py); }});
  ctx.closePath();
  ctx.fillStyle   = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth   = isSelected ? 2.5 : 1.5;
  ctx.fill();
  ctx.stroke();

  if (isSelected) {{
    // Vertex handles
    shape.vertices.forEach(v => {{
      const [vx,vy] = toDisplay(...v);
      ctx.fillStyle   = "#ffffff";
      ctx.strokeStyle = stroke;
      ctx.lineWidth   = 1.5;
      ctx.beginPath(); ctx.arc(vx,vy,HANDLE_RADIUS,0,2*Math.PI);
      ctx.fill(); ctx.stroke();
    }});
    // Rotation handle
    const [rhx,rhy] = rotHandlePos(shape);
    const [cx,cy]   = centroid(shape.vertices);
    const [cdx,cdy] = toDisplay(cx, Math.min(...shape.vertices.map(v=>v[1])));
    ctx.strokeStyle = "#ffd700"; ctx.lineWidth=1;
    ctx.setLineDash([3,2]);
    ctx.beginPath(); ctx.moveTo(cdx,cdy); ctx.lineTo(rhx,rhy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="#ffd700"; ctx.strokeStyle="#fff"; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(rhx,rhy,HANDLE_RADIUS,0,2*Math.PI);
    ctx.fill(); ctx.stroke();
  }}
}}

// ============================================================
// Auto-detect
// ============================================================
async function runDetect() {{
  setStatus("Running detection…");
  const hsvColor   = hexToHsv(document.getElementById("det-color").value);
  const body = {{
    diameter_cm:    parseFloat(document.getElementById("det-diameter").value),
    thickness_cm:   parseFloat(document.getElementById("det-thickness").value),
    expected_count: parseInt(document.getElementById("det-count").value),
    hue_center:     hsvColor.h,
    hue_tolerance:  parseInt(document.getElementById("det-hue-tol").value),
    saturation_min: 80,
    value_max:      parseInt(document.getElementById("det-brightness").value),
  }};
  const res  = await fetch("/masks/detect", {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(body)}});
  const data = await res.json();
  data.polygons.forEach(verts => addShape(verts, maskType, null, true));
  setStatus(`Detection complete: ${{data.count}} circle(s) found.`);
}}

async function saveDetectParams() {{
  const hsvColor = hexToHsv(document.getElementById("det-color").value);
  const params   = {{
    diameter_cm:    parseFloat(document.getElementById("det-diameter").value),
    thickness_cm:   parseFloat(document.getElementById("det-thickness").value),
    expected_count: parseInt(document.getElementById("det-count").value),
    hue_center:     hsvColor.h,
    hue_tolerance:  parseInt(document.getElementById("det-hue-tol").value),
    value_max:      parseInt(document.getElementById("det-brightness").value),
  }};
  await fetch("/project/auto-detect-params", {{method:"POST",
    headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{params}})}});
  setStatus("Auto-detect params saved to project.yaml.");
}}

// ============================================================
// Defaults
// ============================================================
async function loadDefaults() {{
  const res  = await fetch("/masks/defaults");
  const data = await res.json();
  data.shapes.forEach(s => addShape(s.vertices, s.type, null, false));
  setStatus(`Loaded ${{data.shapes.length}} default shape(s).`);
}}

async function saveAsDefaults() {{
  const body = {{shapes: shapes.map(s => ({{type:s.type, vertices:s.vertices}}))}};
  await fetch("/masks/save-defaults", {{method:"POST",
    headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(body)}});
  setStatus("Current shapes saved as project defaults.");
}}

// ============================================================
// Save all
// ============================================================
async function saveAll() {{
  const prefix = "{video_name}";
  let saved = 0;
  for (const shape of shapes) {{
    if (!shape.filename) {{
      const res  = await fetch("/masks/save", {{method:"POST",
        headers:{{"Content-Type":"application/json"}},
        body:JSON.stringify({{vertices:shape.vertices, mask_type:shape.type, prefix}})}});
      const data = await res.json();
      shape.filename = data.filename;
      shape.proposed = false;
      saved++;
    }}
  }}
  refreshShapeList();
  setStatus(`Saved ${{saved}} shape(s). Total: ${{shapes.length}}.`);
}}

// ============================================================
// Utilities
// ============================================================
function setStatus(msg) {{
  document.getElementById("status").textContent = msg;
}}

function hexToHsv(hex) {{
  // Convert hex colour to OpenCV HSV hue (0-179)
  const r = parseInt(hex.slice(1,3),16)/255;
  const g = parseInt(hex.slice(3,5),16)/255;
  const b = parseInt(hex.slice(5,7),16)/255;
  const max=Math.max(r,g,b), min=Math.min(r,g,b), d=max-min;
  let h=0;
  if (d!==0) {{
    if (max===r) h=((g-b)/d)%6;
    else if (max===g) h=(b-r)/d+2;
    else h=(r-g)/d+4;
  }}
  h = Math.round(h*30) % 180;  // OpenCV hue is 0-179
  return {{h: h<0?h+180:h}};
}}
</script>
</body>
</html>"""


# =============================================================================
# Helpers
# =============================================================================

def _read_polygon_csv_raw(polygon_file: Path) -> list:
    """Read a polygon CSV and return [[x, y], ...] as floats."""
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


def _extract_frame(video_file: Path, frame_index: int | None) -> np.ndarray:
    """Extract a single frame from a video file."""
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = frame_index if frame_index is not None else total_frames // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Could not read frame {target_frame} from {video_file}")
    return frame


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive mask editor for TRex tracking regions."
    )
    parser.add_argument("--project", required=True,
                        help="Path to the project folder")
    parser.add_argument("--video",   required=True,
                        help="Video name without extension")
    parser.add_argument("--frame",   type=int, default=None,
                        help="Frame index to display (default: middle frame)")
    parser.add_argument("--port",    type=int, default=8000,
                        help="Local port for the editor (default: 8000)")
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

    frame = _extract_frame(video_file, args.frame)

    # Populate app state before server starts
    APP_STATE["frame"]          = frame
    APP_STATE["video_name"]     = args.video
    APP_STATE["project_dir"]    = project_dir
    APP_STATE["masks_dir"]      = masks_dir
    APP_STATE["meta_real_width"] = meta_real_width

    # Open browser after a short delay to let uvicorn bind
    def open_browser():
        time.sleep(1.2)
        url = f"http://localhost:{args.port}"
        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.info("Open your browser at: %s", url)

    threading.Thread(target=open_browser, daemon=True).start()

    logger.info("Mask editor running at http://localhost:%d", args.port)
    logger.info("Press Ctrl+C to quit.")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
