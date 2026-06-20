"""
_tab_background.py — Background image tab for the unified pipeline app.

Provides:
  - build_background_tab_html(video_name) → HTML fragment for the tab
  - register_background_routes(app, APP_STATE) → FastAPI routes under /background/*

Lets the user generate candidate background images (mean/median/max/min over
n_images frames sampled between start_time/end_time), compare up to 4 at once
in a grid, and save the selected one to 2_pv/average_{video_name}.png.
"""

import base64
import logging
from pathlib import Path

import cv2
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from _background import (
    background_image_path,
    compute_background_image,
    resolve_background_source,
    save_background_image,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic models
# =============================================================================

class GenerateRequest(BaseModel):
    method:     str
    n_images:   int
    start_time: float | None = None
    end_time:   float | None = None


class SaveRequest(BaseModel):
    method:     str
    n_images:   int
    start_time: float | None = None
    end_time:   float | None = None
    overwrite:  bool = False


class SaveParamsRequest(BaseModel):
    params: dict


# =============================================================================
# Route registration
# =============================================================================

def register_background_routes(app: FastAPI, state: dict) -> None:

    @app.get("/background/video-file")
    def get_video_file():
        """
        Serve the current video file directly for the <video> viewer.
        FileResponse supports HTTP range requests natively, so the browser
        can scrub without downloading the whole file first.
        media_type is derived from the actual file extension rather than
        assumed, since video_extension in project.yaml could be anything
        the browser supports (mp4, mov, webm, ...).
        """
        return FileResponse(
            path=state["video_file"],
            media_type=_guess_video_mime_type(state["video_file"]),
        )

    @app.get("/background/info")
    def get_info():
        """Video duration/fps and pre-existing background image (if any)."""
        video_name = state["video_name"]
        pv_dir     = state["pv_dir"]
        fps        = state["fps"]
        total_frames = state["total_frames"]
        duration_s   = total_frames / fps

        existing_image_b64 = None
        existing_file = background_image_path(video_name, pv_dir)
        if existing_file.exists():
            existing_image_b64 = _image_file_to_b64(existing_file)

        return JSONResponse({
            "duration_s":     duration_s,
            "fps":            fps,
            "existing_image": existing_image_b64,
        })

    @app.get("/background/params")
    def get_params():
        """Project-level and (if present) video-level background params."""
        project_config = _read_project_config(state)
        video_name     = state["video_name"]

        project_params = project_config.get("background_params") or {}
        video_overrides = (project_config.get("video_overrides") or {}).get(video_name, {})
        video_params    = video_overrides.get("background_params") or {}

        effective = {**project_params, **video_params}
        return JSONResponse({
            "project_params": project_params,
            "video_params":   video_params,
            "effective":      effective,
            "has_project_defaults": bool(project_params),
        })

    @app.post("/background/params")
    def save_params(request: SaveParamsRequest):
        """Save background generation params as project-level defaults."""
        project_config = _read_project_config(state)
        project_config["background_params"] = request.params
        _write_project_config(state, project_config)
        return JSONResponse({"saved": True})

    @app.post("/background/generate")
    def generate(request: GenerateRequest):
        """Compute a candidate background image; does not save to disk."""
        video_name      = state["video_name"]
        video_extension = state["video_extension"]
        videos_dir      = state["videos_dir"]

        source_file, is_dedicated = resolve_background_source(
            video_name=video_name,
            video_extension=video_extension,
            videos_dir=videos_dir,
        )

        image = compute_background_image(
            video_file=source_file,
            method=request.method,
            n_images=request.n_images,
            start_time=request.start_time,
            end_time=request.end_time,
        )

        _, buffer = cv2.imencode(".png", image)
        b64 = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse({
            "image": b64,
            "used_dedicated_background_video": is_dedicated,
        })

    @app.get("/background/exists")
    def check_exists():
        """Check whether average_{video_name}.png already exists in 2_pv/."""
        existing_file = background_image_path(state["video_name"], state["pv_dir"])
        return JSONResponse({"exists": existing_file.exists()})

    @app.post("/background/save")
    def save(request: SaveRequest):
        """
        Regenerate the image with the given params and save to 2_pv/.
        Also writes per-video override params to project.yaml if they
        differ from the project-level defaults.
        """
        video_name      = state["video_name"]
        video_extension = state["video_extension"]
        videos_dir      = state["videos_dir"]
        pv_dir          = state["pv_dir"]

        output_file = background_image_path(video_name, pv_dir)
        if output_file.exists() and not request.overwrite:
            return JSONResponse({"needs_confirmation": True})

        source_file, _ = resolve_background_source(
            video_name=video_name, video_extension=video_extension, videos_dir=videos_dir,
        )
        image = compute_background_image(
            video_file=source_file,
            method=request.method,
            n_images=request.n_images,
            start_time=request.start_time,
            end_time=request.end_time,
        )
        save_background_image(image, output_file)

        _save_video_override_if_different(state, request)

        return JSONResponse({"saved": True, "path": str(output_file)})


# =============================================================================
# Helpers
# =============================================================================

def _read_project_config(state: dict) -> dict:
    return yaml.safe_load(state["project_yaml_file"].read_text())


def _write_project_config(state: dict, config: dict) -> None:
    state["project_yaml_file"].write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )


def _save_video_override_if_different(state: dict, request: SaveRequest) -> None:
    """
    Compare the saved params against project-level background_params.
    If they differ, write a video_overrides.{video}.background_params entry.
    If identical (or no project-level params set), no override is written.
    """
    project_config = _read_project_config(state)
    video_name     = state["video_name"]
    project_params = project_config.get("background_params") or {}

    used_params = {
        "method":     request.method,
        "n_images":   request.n_images,
        "start_time": request.start_time,
        "end_time":   request.end_time,
    }

    differs = any(
        project_params.get(key) != value
        for key, value in used_params.items()
    )

    if not differs:
        return

    if "video_overrides" not in project_config:
        project_config["video_overrides"] = {}
    if video_name not in project_config["video_overrides"]:
        project_config["video_overrides"][video_name] = {}
    project_config["video_overrides"][video_name]["background_params"] = used_params

    _write_project_config(state, project_config)
    logger.info("Saved video-level background_params override for '%s': %s", video_name, used_params)


VIDEO_MIME_TYPES = {
    ".mp4":  "video/mp4",
    ".m4v":  "video/mp4",
    ".mov":  "video/quicktime",
    ".webm": "video/webm",
    ".avi":  "video/x-msvideo",
    ".mkv":  "video/x-matroska",
}


def _guess_video_mime_type(video_file: Path) -> str:
    """Map a file extension to a MIME type the browser can use for playback."""
    return VIDEO_MIME_TYPES.get(video_file.suffix.lower(), "video/mp4")


def _image_file_to_b64(image_file: Path) -> str:
    image = cv2.imread(str(image_file))
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


# =============================================================================
# HTML fragment
# =============================================================================

def build_background_tab_html(video_name: str) -> str:
    return f"""
<div style="display:flex; width:100%; height:100%;">

  <div id="bg-sidebar" style="width:260px; min-width:260px; background:#16213e;
       display:flex; flex-direction:column; padding:12px; gap:10px;
       overflow-y:auto; border-right:1px solid #0f3460;">

    <h2 style="font-size:13px; color:#a0c4ff; text-transform:uppercase; letter-spacing:1px;">
      Background Image</h2>
    <div id="bg-status" style="font-size:11px; color:#7ec8e3; padding:6px;
         background:#0f3460; border-radius:4px; min-height:32px;">Loading…</div>

    <div style="background:#0f3460; border-radius:6px; padding:10px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">Parameters
        <span id="bg-default-badge" style="display:none; font-size:9px; color:#ffd700;
              background:#3a3000; padding:1px 5px; border-radius:3px; margin-left:6px;">default</span>
      </h3>

      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">Method</label>
        <select id="bg-method" style="width:100%; padding:4px 6px; background:#1a1a2e;
                border:1px solid #415a77; border-radius:4px; color:#e0e0e0; font-size:12px;">
          <option value="mean">mean</option>
          <option value="median" selected>median</option>
          <option value="max">max</option>
          <option value="min">min</option>
        </select>
      </div>

      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">Number of images</label>
        <input type="number" id="bg-n-images" min="1" value="10"
               style="width:100%; padding:4px 6px; background:#1a1a2e;
               border:1px solid #415a77; border-radius:4px; color:#e0e0e0; font-size:12px;">
      </div>

      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">Start time (s, optional)</label>
        <input type="number" id="bg-start-time" step="0.1" placeholder="auto (video start + 1s)"
               style="width:100%; padding:4px 6px; background:#1a1a2e;
               border:1px solid #415a77; border-radius:4px; color:#e0e0e0; font-size:12px;">
      </div>

      <div style="margin-bottom:6px;">
        <label style="font-size:11px; color:#aaa; display:block; margin-bottom:2px;">End time (s, optional)</label>
        <input type="number" id="bg-end-time" step="0.1" placeholder="auto (video end - 1s)"
               style="width:100%; padding:4px 6px; background:#1a1a2e;
               border:1px solid #415a77; border-radius:4px; color:#e0e0e0; font-size:12px;">
      </div>

      <button onclick="bgGenerate()" style="width:100%; padding:7px 10px; border:none;
              border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
              background:#e76f51; color:#fff;">🔄 Generate</button>
      <button onclick="bgSaveParams()" style="width:100%; margin-top:5px; padding:7px 10px;
              border:none; border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
              background:#415a77; color:#e0e0e0;">💾 Save params to project</button>
    </div>

    <div style="font-size:10px; color:#789; line-height:1.5;">
      Click an image in the grid to select it. Del removes the selected image
      from the grid (memory only). Ctrl+S saves the selected image to the
      project's 2_pv folder.
    </div>

    <button onclick="bgSaveSelected()" style="width:100%; margin-top:auto; padding:7px 10px;
            border:none; border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
            background:#1b4332; color:#d8f3dc;" title="Ctrl+S">
      💾 Save selected as background (Ctrl+S)
    </button>
  </div>

  <div style="flex:1; display:flex; align-items:center; justify-content:center;
       padding:16px; overflow:auto;">
    <div id="bg-grid" style="display:grid; grid-template-columns:1fr 1fr;
         gap:12px; width:100%; height:100%; max-width:1100px;">
      <div id="bg-video-cell" style="position:relative; border:2px solid #333;
           border-radius:6px; overflow:hidden; display:flex; align-items:center;
           justify-content:center; background:#000;">
        <video id="bg-video-viewer" controls style="max-width:100%; max-height:100%;
               width:100%; height:100%; object-fit:contain;"></video>
        <span style="position:absolute; top:4px; left:4px; font-size:10px;
              background:rgba(0,0,0,0.7); color:#7ec8e3; padding:2px 6px;
              border-radius:3px;">video</span>
      </div>
      <div id="bg-candidates"
           style="display:contents;"></div>
    </div>
  </div>

  <div id="bg-overwrite-overlay" style="display:none; position:fixed; inset:0;
       background:rgba(0,0,0,0.75); align-items:center; justify-content:center; z-index:150;">
    <div style="background:#16213e; border:1px solid #a0c4ff; border-radius:10px;
         padding:28px 32px; text-align:center; max-width:340px;">
      <h3 style="color:#a0c4ff; margin-bottom:14px;">Overwrite existing background image?</h3>
      <p style="color:#ccc; font-size:13px; margin-bottom:20px;">
        A background image already exists for this video.</p>
      <div style="display:flex; gap:10px; justify-content:center;">
        <button onclick="bgConfirmOverwrite()" style="width:auto; padding:8px 20px; border:none;
                border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
                background:#9d0208; color:#ffccd5;">Overwrite</button>
        <button onclick="bgCancelOverwrite()" style="width:auto; padding:8px 20px; border:none;
                border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
                background:#415a77; color:#e0e0e0;">Cancel</button>
      </div>
    </div>
  </div>
</div>

<script>
(function() {{
  const VIDEO_NAME = "{video_name}";
  let bgCandidates = [];   // {{ id, image_b64, params, isPreExisting }}
  let bgSelectedId = null;
  let bgNextId = 1;
  let bgProjectDefaults = {{}};
  let bgPendingSave = null;

  window.bgInit = async function() {{
    document.getElementById("bg-video-viewer").src = "/background/video-file";
    await bgLoadInfo();
    await bgLoadParams();
  }};

  async function bgLoadInfo() {{
    const res = await fetch("/background/info");
    const d = await res.json();
    if (d.existing_image) {{
      bgAddCandidate(d.existing_image, {{}}, true);
    }}
    bgSetStatus(`Video duration: ${{d.duration_s.toFixed(1)}}s @ ${{d.fps.toFixed(1)}}fps`);
  }}

  async function bgLoadParams() {{
    const res = await fetch("/background/params");
    const d = await res.json();
    bgProjectDefaults = d.project_params;
    const eff = d.effective;
    if (eff.method)     document.getElementById("bg-method").value = eff.method;
    if (eff.n_images)    document.getElementById("bg-n-images").value = eff.n_images;
    if (eff.start_time != null) document.getElementById("bg-start-time").value = eff.start_time;
    if (eff.end_time   != null) document.getElementById("bg-end-time").value   = eff.end_time;
    document.getElementById("bg-default-badge").style.display =
      (d.has_project_defaults && !Object.keys(d.video_params).length) ? "inline" : "none";
  }}

  function bgCurrentParams() {{
    const startVal = document.getElementById("bg-start-time").value;
    const endVal   = document.getElementById("bg-end-time").value;
    return {{
      method:     document.getElementById("bg-method").value,
      n_images:   parseInt(document.getElementById("bg-n-images").value),
      start_time: startVal === "" ? null : parseFloat(startVal),
      end_time:   endVal   === "" ? null : parseFloat(endVal),
    }};
  }}

  window.bgGenerate = async function() {{
    if (bgCandidates.length >= 3) {{
      bgSetStatus("3 candidates max. Delete one first (select + Del).");
      return;
    }}
    bgSetStatus("Generating…");
    const params = bgCurrentParams();
    const res = await fetch("/background/generate", {{method:"POST",
      headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(params)}});
    const d = await res.json();
    bgAddCandidate(d.image, params, false);
    bgSetStatus(d.used_dedicated_background_video
      ? "Generated from dedicated background video."
      : "Generated from main tracking video.");
  }};

  function bgAddCandidate(imageB64, params, isPreExisting) {{
    const id = bgNextId++;
    bgCandidates.push({{ id, image: imageB64, params, isPreExisting }});
    bgSelectedId = id;
    bgRenderGrid();
  }}

  function bgRenderGrid() {{
    const grid = document.getElementById("bg-candidates");
    grid.innerHTML = "";
    bgCandidates.forEach(c => {{
      const cell = document.createElement("div");
      cell.style.position = "relative";
      cell.style.border = (c.id === bgSelectedId) ? "3px solid #a0c4ff" : "2px solid #333";
      cell.style.borderRadius = "6px";
      cell.style.overflow = "hidden";
      cell.style.cursor = "pointer";
      cell.style.display = "flex";
      cell.style.alignItems = "center";
      cell.style.justifyContent = "center";
      cell.style.background = "#000";
      cell.onclick = () => {{ bgSelectedId = c.id; bgRenderGrid(); bgShowParams(c); }};

      const img = document.createElement("img");
      img.src = "data:image/png;base64," + c.image;
      img.style.maxWidth = "100%"; img.style.maxHeight = "100%"; img.style.objectFit = "contain";
      cell.appendChild(img);

      if (c.isPreExisting) {{
        const label = document.createElement("span");
        label.textContent = "pre-existing";
        label.style.position = "absolute"; label.style.top = "4px"; label.style.left = "4px";
        label.style.fontSize = "10px"; label.style.background = "rgba(0,0,0,0.7)";
        label.style.color = "#ffd700"; label.style.padding = "2px 6px"; label.style.borderRadius = "3px";
        cell.appendChild(label);
      }}
      grid.appendChild(cell);
    }});
  }}

  function bgShowParams(candidate) {{
    if (!candidate.params || !Object.keys(candidate.params).length) return;
    const p = candidate.params;
    if (p.method)   document.getElementById("bg-method").value = p.method;
    if (p.n_images) document.getElementById("bg-n-images").value = p.n_images;
    document.getElementById("bg-start-time").value = p.start_time ?? "";
    document.getElementById("bg-end-time").value   = p.end_time   ?? "";
  }}

  window.bgSaveParams = async function() {{
    const params = bgCurrentParams();
    await fetch("/background/params", {{method:"POST",
      headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{params}})}});
    bgSetStatus("Parameters saved to project.yaml.");
  }};

  window.bgSaveSelected = async function() {{
    const candidate = bgCandidates.find(c => c.id === bgSelectedId);
    if (!candidate) {{ bgSetStatus("No image selected."); return; }}
    const params = candidate.params && Object.keys(candidate.params).length
      ? candidate.params : bgCurrentParams();
    bgPendingSave = params;
    const res = await fetch("/background/save", {{method:"POST",
      headers:{{"Content-Type":"application/json"}}, body:JSON.stringify({{...params, overwrite:false}})}});
    const d = await res.json();
    if (d.needs_confirmation) {{
      document.getElementById("bg-overwrite-overlay").style.display = "flex";
    }} else {{
      bgSetStatus("Saved as background image.");
      bgPendingSave = null;
    }}
  }};

  window.bgConfirmOverwrite = async function() {{
    document.getElementById("bg-overwrite-overlay").style.display = "none";
    await fetch("/background/save", {{method:"POST",
      headers:{{"Content-Type":"application/json"}},
      body:JSON.stringify({{...bgPendingSave, overwrite:true}})}});
    bgSetStatus("Background image overwritten.");
    bgPendingSave = null;
  }};

  window.bgCancelOverwrite = function() {{
    document.getElementById("bg-overwrite-overlay").style.display = "none";
    bgPendingSave = null;
  }};

  function bgSetStatus(msg) {{ document.getElementById("bg-status").textContent = msg; }}

  document.addEventListener("keydown", e => {{
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (activeTab !== "background") return;
    if (e.ctrlKey && (e.key === "s" || e.key === "S")) {{ e.preventDefault(); bgSaveSelected(); }}
    if ((e.key === "Delete" || e.key === "Backspace") && bgSelectedId !== null) {{
      bgCandidates = bgCandidates.filter(c => c.id !== bgSelectedId);
      bgSelectedId = bgCandidates.length ? bgCandidates[bgCandidates.length-1].id : null;
      bgRenderGrid();
    }}
  }});

  // Background tab is rendered once per page load (not lazily), so initialise
  // immediately rather than waiting for a tab-activation event.
  window.bgInit();
}})();
</script>
"""
