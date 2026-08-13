"""
_tab_batches.py — Batches tab for the unified pipeline app.

Lets the user:
  - Group videos into named batches (shift-click range select in the video list)
  - Assign/remove videos from batches (a video may belong to at most one batch)
  - Set batch-level description, confirmed_detect_threshold, video_conversion_range
  - View a thumbnail of the batch's mask shapes (rendered from the first
    member video's mask files, overlaid on a neutral grid background)

Batches are stored directly in project.yaml under `batches:` — no separate
files. Mask thumbnails are rendered on demand (not cached) since mask edits
are infrequent and project videos are small enough for this to be fast.
"""

import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = (220, 160)   # width, height in px


# =============================================================================
# Pydantic models
# =============================================================================

class AssignRequest(BaseModel):
    video_names: list[str]
    batch_name:  str          # created if it does not exist yet

class RemoveRequest(BaseModel):
    video_name: str
    batch_name: str

class UpdateBatchRequest(BaseModel):
    batch_name: str
    description: str | None = None
    confirmed_detect_threshold: int | None = None
    video_conversion_range: list[int] | None = None
    track_max_individuals: int | None = None
    individual_prefix: str | None = None

class UpdateProjectParamsRequest(BaseModel):
    meta_real_width:       float | None = None
    track_max_individuals: int | None = None
    individual_prefix:     str | None = None
    video_extension:       str | None = None


# =============================================================================
# Route registration
# =============================================================================

def register_batches_routes(app: FastAPI, state: dict) -> None:

    @app.get("/batches")
    def get_batches():
        """
        Return all videos (with their batch membership, if any) and the
        full batch structure (description, settings, members).
        """
        project_config = _read_project_config(state)
        videos_dir, _, _ = _resolve_dirs(state)
        video_extension = state["video_extension"]
        all_videos = sorted(
            video_file.stem for video_file in videos_dir.glob(f"*.{video_extension}")
        )

        batches_config = project_config.get("batches") or {}
        video_to_batch = {}
        for batch_name, batch_config in batches_config.items():
            for video_name in (batch_config.get("videos") or []):
                video_to_batch[video_name] = batch_name

        videos = [
            {"video_name": v, "batch_name": video_to_batch.get(v)}
            for v in all_videos
        ]

        batches = [
            {
                "batch_name":                 batch_name,
                "description":                batch_config.get("description", ""),
                "confirmed_detect_threshold": batch_config.get("confirmed_detect_threshold"),
                "video_conversion_range":     batch_config.get("video_conversion_range"),
                "track_max_individuals":      batch_config.get("track_max_individuals"),
                "individual_prefix":          batch_config.get("individual_prefix"),
                "videos":                     batch_config.get("videos") or [],
            }
            for batch_name, batch_config in batches_config.items()
        ]

        background_params = project_config.get("background_params") or {}

        return JSONResponse({
            "videos":                videos,
            "batches":               batches,
            "background_params":    background_params,
            "meta_real_width":       project_config.get("meta_real_width"),
            "track_max_individuals": project_config.get("track_max_individuals"),
            "individual_prefix":     project_config.get("individual_prefix"),
            "video_extension":       project_config.get("video_extension"),
        })

    @app.post("/batches/assign")
    def assign(request: AssignRequest):
        """
        Add video_names to batch_name (creating it if needed). Each video is
        first removed from any other batch it currently belongs to, so a
        video never ends up in more than one batch.
        """
        project_config = _read_project_config(state)
        if "batches" not in project_config:
            project_config["batches"] = {}
        batches_config = project_config["batches"]

        for batch_name, batch_config in batches_config.items():
            if batch_config.get("videos"):
                batch_config["videos"] = [
                    v for v in batch_config["videos"] if v not in request.video_names
                ]

        if request.batch_name not in batches_config:
            batches_config[request.batch_name] = {"videos": []}
        if "videos" not in batches_config[request.batch_name]:
            batches_config[request.batch_name]["videos"] = []

        existing = set(batches_config[request.batch_name]["videos"])
        for video_name in request.video_names:
            existing.add(video_name)
        batches_config[request.batch_name]["videos"] = sorted(existing)

        _write_project_config(state, project_config)
        logger.info(
            "Assigned %s to batch '%s'.", request.video_names, request.batch_name
        )
        return JSONResponse({"ok": True})

    @app.post("/batches/remove")
    def remove(request: RemoveRequest):
        """Remove one video from a batch (it becomes unassigned)."""
        project_config = _read_project_config(state)
        batches_config = project_config.get("batches") or {}
        batch_config   = batches_config.get(request.batch_name)
        if batch_config and request.video_name in (batch_config.get("videos") or []):
            batch_config["videos"].remove(request.video_name)
            _write_project_config(state, project_config)
        return JSONResponse({"ok": True})

    @app.delete("/batches/{batch_name}")
    def delete_batch(batch_name: str):
        """Delete a batch entirely. Its videos become unassigned."""
        project_config = _read_project_config(state)
        batches_config = project_config.get("batches") or {}
        if batch_name in batches_config:
            del batches_config[batch_name]
            _write_project_config(state, project_config)
        return JSONResponse({"ok": True})

    @app.post("/batches/update")
    def update_batch(request: UpdateBatchRequest):
        """
        Update a batch's description / confirmed_detect_threshold /
        video_conversion_range / track_max_individuals / individual_prefix.
        """
        project_config = _read_project_config(state)
        batches_config = project_config.get("batches") or {}
        if request.batch_name not in batches_config:
            return JSONResponse({"error": f"Batch '{request.batch_name}' not found"}, status_code=404)

        batch_config = batches_config[request.batch_name]
        if request.description is not None:
            batch_config["description"] = request.description
        if request.confirmed_detect_threshold is not None:
            batch_config["confirmed_detect_threshold"] = request.confirmed_detect_threshold
        if request.video_conversion_range is not None:
            batch_config["video_conversion_range"] = request.video_conversion_range
        if request.track_max_individuals is not None:
            batch_config["track_max_individuals"] = request.track_max_individuals
        if request.individual_prefix is not None:
            batch_config["individual_prefix"] = request.individual_prefix

        _write_project_config(state, project_config)
        return JSONResponse({"ok": True})

    @app.post("/batches/update-project-params")
    def update_project_params(request: UpdateProjectParamsRequest):
        """Update project-level meta_real_width / track_max_individuals / individual_prefix."""
        project_config = _read_project_config(state)
        if request.meta_real_width is not None:
            project_config["meta_real_width"] = request.meta_real_width
        if request.track_max_individuals is not None:
            project_config["track_max_individuals"] = request.track_max_individuals
        if request.individual_prefix is not None:
            project_config["individual_prefix"] = request.individual_prefix
        if request.video_extension is not None:
            project_config["video_extension"] = request.video_extension

        _write_project_config(state, project_config)
        logger.info(
            "Updated project params: meta_real_width=%s, track_max_individuals=%s, individual_prefix=%s",
            request.meta_real_width, request.track_max_individuals, request.individual_prefix,
        )
        return JSONResponse({"ok": True})

    @app.get("/batches/{batch_name}/mask-thumbnail")
    def mask_thumbnail(batch_name: str):
        """
        Render a small preview of the batch's mask shapes on a neutral grid
        background. Returns null image if the batch has no masks.
        """
        _, _, masks_dir = _resolve_dirs(state)
        image_b64 = _render_mask_thumbnail(batch_name, masks_dir)
        return JSONResponse({"image": image_b64})


# =============================================================================
# Helpers
# =============================================================================

def _resolve_dirs(state: dict) -> tuple[Path, Path, Path]:
    """
    Mirror app.py's _base_dirs logic locally (videos_dir, pv_dir, masks_dir)
    so this module does not need to import from app.py directly.
    """
    root = state["project_dir"] / "2_tracking"
    if state.get("mode") == "tuning":
        tuning_dir = root / "tuning"
        return tuning_dir / "1_videos", tuning_dir / "2_pv", tuning_dir / "masks"
    return root / "1_videos", root / "2_pv", root / "masks"


def _read_project_config(state: dict) -> dict:
    return yaml.safe_load(state["project_yaml_file"].read_text())


def _write_project_config(state: dict, config: dict) -> None:
    state["project_yaml_file"].write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )


def _render_mask_thumbnail(batch_name: str, masks_dir: Path) -> str | None:
    """
    Draw batch-level mask polygons on a neutral grid background, scaled to
    THUMBNAIL_SIZE. Returns a base64 PNG, or None if no masks exist.
    """
    from ._masks import load_masks_for_batch

    masks = load_masks_for_batch(batch_name, masks_dir)
    if not masks["include"] and not masks["ignore"]:
        return None

    width, height = THUMBNAIL_SIZE
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    grid_spacing = 20
    for x in range(0, width, grid_spacing):
        cv2.line(canvas, (x, 0), (x, height), (45, 45, 45), 1)
    for y in range(0, height, grid_spacing):
        cv2.line(canvas, (0, y), (width, y), (45, 45, 45), 1)

    all_points = [
        point
        for polygon in (masks["include"] + masks["ignore"])
        for point in polygon
    ]
    if not all_points:
        return None

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1)
    span_y = max(max_y - min_y, 1)

    padding = 10
    scale = min(
        (width  - 2 * padding) / span_x,
        (height - 2 * padding) / span_y,
    )

    def to_canvas(point):
        x, y = point
        return (
            int(padding + (x - min_x) * scale),
            int(padding + (y - min_y) * scale),
        )

    for polygon in masks["include"]:
        points = np.array([to_canvas(p) for p in polygon], dtype=np.int32)
        cv2.fillPoly(canvas, [points], (40, 90, 50))
        cv2.polylines(canvas, [points], True, (50, 220, 80), 1)

    for polygon in masks["ignore"]:
        points = np.array([to_canvas(p) for p in polygon], dtype=np.int32)
        cv2.fillPoly(canvas, [points], (40, 40, 90))
        cv2.polylines(canvas, [points], True, (50, 60, 220), 1)

    _, buffer = cv2.imencode(".png", canvas)
    return base64.b64encode(buffer).decode("utf-8")


# =============================================================================
# HTML fragment
# =============================================================================

def build_batches_tab_html(video_name: str) -> str:
    return f"""
<div style="display:flex; width:100%; height:100%;">

  <div style="width:280px; min-width:280px; background:#16213e;
       display:flex; flex-direction:column; padding:12px; gap:10px;
       overflow-y:auto; border-right:1px solid #0f3460;">
    <h2 style="font-size:13px; color:#a0c4ff; text-transform:uppercase; letter-spacing:1px;">
      Videos</h2>
    <div style="font-size:10px; color:#789;">
      Click to select, Shift+click for a range. Selected videos can be
      added to a batch below.
    </div>
    <div id="batch-video-list" style="flex:1; overflow-y:auto;"></div>

    <div style="background:#0f3460; border-radius:6px; padding:10px;">
      <label style="font-size:11px; color:#aaa; display:block; margin-bottom:4px;">
        Add selected to batch</label>
      <select id="batch-target-select" style="width:100%; padding:5px 6px; background:#1a1a2e;
              border:1px solid #415a77; border-radius:4px; color:#e0e0e0; font-size:12px;
              margin-bottom:6px;">
        <option value="__new__">+ New batch…</option>
      </select>
      <input type="text" id="batch-new-name" placeholder="New batch name"
             style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
             border-radius:4px; color:#e0e0e0; font-size:12px; margin-bottom:6px; display:none;">
      <button onclick="batchAssignSelected()" style="width:100%; padding:7px 10px; border:none;
              border-radius:5px; cursor:pointer; font-size:12px; font-weight:600;
              background:#2d6a4f; color:#d8f3dc;">+ Add to batch</button>
    </div>
  </div>

  <div style="flex:1; overflow-y:auto; padding:16px;">
    <div id="batch-status" style="font-size:11px; color:#7ec8e3; padding:6px;
         background:#0f3460; border-radius:4px; margin-bottom:12px; min-height:20px;"></div>

    <div style="background:#16213e; border:1px solid #0f3460; border-radius:8px;
         padding:12px 14px; margin-bottom:14px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">Project parameters</h3>
      <div style="display:flex; gap:10px;">
        <div style="flex:1;">
          <label style="font-size:11px; color:#aaa; display:block; margin-bottom:3px;">
            Arena width (cm)</label>
          <input type="number" id="proj-meta-real-width" step="0.1"
                 style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;"
                 onblur="saveProjectParams()">
        </div>
        <div style="flex:1;">
          <label style="font-size:11px; color:#aaa; display:block; margin-bottom:3px;">
            Max individuals</label>
          <input type="number" id="proj-max-individuals" min="1"
                 style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;"
                 onblur="saveProjectParams()">
        </div>
        <div style="flex:1;">
          <label style="font-size:11px; color:#aaa; display:block; margin-bottom:3px;">
            Individual prefix</label>
          <input type="text" id="proj-individual-prefix"
                 style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;"
                 onblur="saveProjectParams()">
        </div>
        <div style="flex:1;">
          <label style="font-size:11px; color:#aaa; display:block; margin-bottom:3px;">
            Video extension <span style="color:#5a7a99;">(override)</span></label>
          <input type="text" id="proj-video-extension" placeholder="auto"
                 style="width:100%; padding:5px 6px; background:#1a1a2e; border:1px solid #415a77;
                 border-radius:4px; color:#e0e0e0; font-size:12px;"
                 onblur="saveProjectParams()">
        </div>
      </div>
    </div>

    <div style="background:#16213e; border:1px solid #0f3460; border-radius:8px;
         padding:12px 14px; margin-bottom:14px;">
      <h3 style="font-size:11px; color:#7ec8e3; text-transform:uppercase;
           letter-spacing:0.8px; margin-bottom:8px;">Project background defaults</h3>
      <div id="batch-bg-params" style="font-size:12px; color:#ccc;">Loading…</div>
    </div>

    <div id="batch-list" style="display:flex; flex-direction:column; gap:12px;"></div>
  </div>
</div>

<script>
(function() {{
  let batchVideos = [];      // [{{video_name, batch_name}}]
  let batchBatches = [];     // [{{batch_name, description, confirmed_detect_threshold, video_conversion_range, videos}}]
  let batchSelected = new Set();
  let batchLastClickedIndex = null;

  window.batchesInit = async function() {{
    await batchLoad();
  }};

  async function batchLoad() {{
    const res = await fetch("/batches"), d = await res.json();
    batchVideos  = d.videos;
    batchBatches = d.batches;
    renderVideoList();
    renderTargetSelect();
    renderBatchList();
    renderBackgroundParams(d.background_params || {{}});
    renderProjectParams(d);
  }}

  function renderProjectParams(d) {{
    document.getElementById("proj-meta-real-width").value   = d.meta_real_width ?? "";
    document.getElementById("proj-max-individuals").value   = d.track_max_individuals ?? "";
    document.getElementById("proj-individual-prefix").value = d.individual_prefix ?? "";
    document.getElementById("proj-video-extension").value   = d.video_extension ?? "";
  }}

  window.saveProjectParams = async function() {{
    const widthRaw   = document.getElementById("proj-meta-real-width").value;
    const maxRaw     = document.getElementById("proj-max-individuals").value;
    const prefixRaw  = document.getElementById("proj-individual-prefix").value;
    const extRaw     = document.getElementById("proj-video-extension").value.trim();
    const body = {{
      meta_real_width:       widthRaw === "" ? null : parseFloat(widthRaw),
      track_max_individuals: maxRaw === ""   ? null : parseInt(maxRaw),
      individual_prefix:     prefixRaw === "" ? null : prefixRaw,
      video_extension:       extRaw === "" ? null : extRaw,
    }};
    await fetch("/batches/update-project-params", {{method:"POST",
      headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(body)}});
    batchSetStatus("Project parameters saved.");
  }};

  function renderBackgroundParams(params) {{
    const container = document.getElementById("batch-bg-params");
    if (!params || Object.keys(params).length === 0) {{
      container.textContent = "Not set — using default.settings fallback (background video if present, otherwise computed by TGrabs).";
      return;
    }}
    const rows = [
      ["Method",     params.method ?? "—"],
      ["N images",   params.n_images ?? "—"],
      ["Start time", params.start_time ?? "auto (video start + 1s)"],
      ["End time",   params.end_time ?? "auto (video end - 1s)"],
    ];
    container.innerHTML = rows.map(([label, value]) =>
      `<div style="display:flex; gap:8px;"><span style="color:#789; width:90px;">${{label}}:</span><span>${{value}}</span></div>`
    ).join("");
  }}

  function renderVideoList() {{
    const list = document.getElementById("batch-video-list");
    list.innerHTML = "";
    batchVideos.forEach((v, index) => {{
      const row = document.createElement("div");
      row.style.display = "flex"; row.style.alignItems = "center";
      row.style.gap = "6px"; row.style.padding = "4px 6px";
      row.style.borderRadius = "4px"; row.style.cursor = "pointer";
      row.style.fontSize = "12px";
      row.style.background = batchSelected.has(v.video_name) ? "#1a3a5c" : "transparent";
      row.onmouseenter = () => {{ if (!batchSelected.has(v.video_name)) row.style.background = "#12233f"; }};
      row.onmouseleave = () => {{ if (!batchSelected.has(v.video_name)) row.style.background = "transparent"; }};
      row.onclick = (e) => onVideoRowClick(e, v.video_name, index);

      const checkbox = document.createElement("span");
      checkbox.textContent = batchSelected.has(v.video_name) ? "☑" : "☐";
      checkbox.style.width = "16px";

      const label = document.createElement("span");
      label.style.flex = "1";
      label.textContent = v.video_name;

      row.appendChild(checkbox);
      row.appendChild(label);

      if (v.batch_name) {{
        const tag = document.createElement("span");
        tag.textContent = v.batch_name;
        tag.style.fontSize = "9px"; tag.style.background = "#3a3000";
        tag.style.color = "#ffd700"; tag.style.padding = "1px 5px";
        tag.style.borderRadius = "3px";
        row.appendChild(tag);
      }}
      list.appendChild(row);
    }});
  }}

  function onVideoRowClick(e, videoName, index) {{
    if (e.shiftKey && batchLastClickedIndex !== null) {{
      const [start, end] = [batchLastClickedIndex, index].sort((a,b)=>a-b);
      for (let i = start; i <= end; i++) {{
        batchSelected.add(batchVideos[i].video_name);
      }}
    }} else {{
      if (batchSelected.has(videoName)) batchSelected.delete(videoName);
      else batchSelected.add(videoName);
      batchLastClickedIndex = index;
    }}
    renderVideoList();
  }}

  function renderTargetSelect() {{
    const select = document.getElementById("batch-target-select");
    const newNameInput = document.getElementById("batch-new-name");
    select.innerHTML = "";
    batchBatches.forEach(b => {{
      const opt = document.createElement("option");
      opt.value = b.batch_name; opt.textContent = b.batch_name;
      select.appendChild(opt);
    }});
    const newOpt = document.createElement("option");
    newOpt.value = "__new__"; newOpt.textContent = "+ New batch…";
    select.appendChild(newOpt);
    select.onchange = () => {{
      newNameInput.style.display = (select.value === "__new__") ? "" : "none";
    }};
    newNameInput.style.display = (select.value === "__new__") ? "" : "none";
  }}

  window.batchAssignSelected = async function() {{
    if (batchSelected.size === 0) {{ batchSetStatus("No videos selected."); return; }}
    const select = document.getElementById("batch-target-select");
    let batchName = select.value;
    if (batchName === "__new__") {{
      batchName = document.getElementById("batch-new-name").value.trim();
      if (!batchName) {{ batchSetStatus("Enter a name for the new batch."); return; }}
    }}
    await fetch("/batches/assign", {{method:"POST", headers:{{"Content-Type":"application/json"}},
      body: JSON.stringify({{video_names: Array.from(batchSelected), batch_name: batchName}})}});
    batchSetStatus(`Added ${{batchSelected.size}} video(s) to '${{batchName}}'.`);
    batchSelected.clear();
    document.getElementById("batch-new-name").value = "";
    await batchLoad();
  }};

  function renderBatchList() {{
    const container = document.getElementById("batch-list");
    container.innerHTML = "";
    batchBatches.forEach(b => container.appendChild(renderBatchCard(b)));

    const addCard = document.createElement("div");
    addCard.style.padding = "10px"; addCard.style.fontSize = "11px"; addCard.style.color = "#5a7a99";
    addCard.textContent = batchBatches.length === 0
      ? "No batches yet — select videos on the left and add them to a new batch."
      : "";
    container.appendChild(addCard);
  }}

  function renderBatchCard(b) {{
    const card = document.createElement("div");
    card.style.background = "#16213e"; card.style.border = "1px solid #0f3460";
    card.style.borderRadius = "8px"; card.style.padding = "14px";

    const header = document.createElement("div");
    header.style.display = "flex"; header.style.justifyContent = "space-between";
    header.style.alignItems = "center"; header.style.marginBottom = "8px";
    const title = document.createElement("h3");
    title.textContent = b.batch_name;
    title.style.fontSize = "13px"; title.style.color = "#a0c4ff";
    const delBtn = document.createElement("button");
    delBtn.textContent = "✕ Delete batch";
    delBtn.style.background = "none"; delBtn.style.border = "none";
    delBtn.style.color = "#ff6b6b"; delBtn.style.cursor = "pointer"; delBtn.style.fontSize = "11px";
    delBtn.onclick = async () => {{
      if (!confirm(`Delete batch '${{b.batch_name}}'? Videos become unassigned.`)) return;
      await fetch(`/batches/${{b.batch_name}}`, {{method:"DELETE"}});
      await batchLoad();
    }};
    header.appendChild(title); header.appendChild(delBtn);
    card.appendChild(header);

    const body = document.createElement("div");
    body.style.display = "flex"; body.style.gap = "14px";

    // Mask thumbnail
    const thumbWrap = document.createElement("div");
    thumbWrap.style.width = "220px"; thumbWrap.style.height = "160px";
    thumbWrap.style.background = "#000"; thumbWrap.style.borderRadius = "5px";
    thumbWrap.style.display = "flex"; thumbWrap.style.alignItems = "center";
    thumbWrap.style.justifyContent = "center"; thumbWrap.style.flexShrink = "0";
    thumbWrap.style.fontSize = "10px"; thumbWrap.style.color = "#445";
    thumbWrap.textContent = "Loading mask preview…";
    fetch(`/batches/${{b.batch_name}}/mask-thumbnail`).then(r => r.json()).then(d => {{
      if (d.image) {{
        thumbWrap.textContent = "";
        const img = document.createElement("img");
        img.src = "data:image/png;base64," + d.image;
        img.style.maxWidth = "100%"; img.style.maxHeight = "100%";
        thumbWrap.appendChild(img);
      }} else {{
        thumbWrap.textContent = "No masks for this batch";
      }}
    }});
    body.appendChild(thumbWrap);

    // Fields
    const fields = document.createElement("div");
    fields.style.flex = "1"; fields.style.display = "flex";
    fields.style.flexDirection = "column"; fields.style.gap = "8px";

    fields.appendChild(makeField("Description", "textarea", b.description || "", (val) => {{
      batchUpdate(b.batch_name, {{description: val}});
    }}));

    const thresholdRangeRow = document.createElement("div");
    thresholdRangeRow.style.display = "flex"; thresholdRangeRow.style.gap = "10px";
    thresholdRangeRow.appendChild(makeField("Confirmed threshold", "number",
      b.confirmed_detect_threshold ?? "", (val) => {{
        batchUpdate(b.batch_name, {{confirmed_detect_threshold: val === "" ? null : parseInt(val)}});
      }}, true));
    thresholdRangeRow.appendChild(makeField("Conversion range [start,end]", "text",
      b.video_conversion_range ? b.video_conversion_range.join(",") : "", (val) => {{
        const parts = val.split(",").map(s => parseInt(s.trim()));
        if (parts.length === 2 && !parts.some(isNaN)) {{
          batchUpdate(b.batch_name, {{video_conversion_range: parts}});
        }}
      }}, true));
    fields.appendChild(thresholdRangeRow);

    const individualsPrefixRow = document.createElement("div");
    individualsPrefixRow.style.display = "flex"; individualsPrefixRow.style.gap = "10px";
    individualsPrefixRow.appendChild(makeField("Max individuals", "number",
      b.track_max_individuals ?? "", (val) => {{
        batchUpdate(b.batch_name, {{track_max_individuals: val === "" ? null : parseInt(val)}});
      }}, true));
    individualsPrefixRow.appendChild(makeField("Individual prefix", "text",
      b.individual_prefix ?? "", (val) => {{
        batchUpdate(b.batch_name, {{individual_prefix: val === "" ? null : val}});
      }}, true));
    fields.appendChild(individualsPrefixRow);

    const membersLabel = document.createElement("div");
    membersLabel.style.fontSize = "11px"; membersLabel.style.color = "#aaa"; membersLabel.style.marginTop = "4px";
    membersLabel.textContent = `Members (${{b.videos.length}})`;
    fields.appendChild(membersLabel);

    const membersList = document.createElement("div");
    membersList.style.display = "flex"; membersList.style.flexWrap = "wrap"; membersList.style.gap = "5px";
    b.videos.forEach(videoName => {{
      const chip = document.createElement("span");
      chip.style.fontSize = "11px"; chip.style.background = "#0f3460";
      chip.style.padding = "3px 8px"; chip.style.borderRadius = "10px";
      chip.style.display = "flex"; chip.style.alignItems = "center"; chip.style.gap = "5px";
      chip.textContent = videoName;
      const removeBtn = document.createElement("span");
      removeBtn.textContent = "✕"; removeBtn.style.cursor = "pointer"; removeBtn.style.color = "#ff6b6b";
      removeBtn.onclick = async () => {{
        await fetch("/batches/remove", {{method:"POST", headers:{{"Content-Type":"application/json"}},
          body: JSON.stringify({{video_name: videoName, batch_name: b.batch_name}})}});
        await batchLoad();
      }};
      chip.appendChild(removeBtn);
      membersList.appendChild(chip);
    }});
    fields.appendChild(membersList);

    body.appendChild(fields);
    card.appendChild(body);
    return card;
  }}

  function makeField(labelText, type, value, onChange, compact=false) {{
    const wrap = document.createElement("div");
    wrap.style.flex = compact ? "1" : "";
    const label = document.createElement("label");
    label.textContent = labelText;
    label.style.fontSize = "11px"; label.style.color = "#aaa";
    label.style.display = "block"; label.style.marginBottom = "3px";
    wrap.appendChild(label);

    let input;
    if (type === "textarea") {{
      input = document.createElement("textarea");
      input.rows = 2;
    }} else {{
      input = document.createElement("input");
      input.type = type;
    }}
    input.value = value;
    input.style.width = "100%"; input.style.padding = "5px 6px";
    input.style.background = "#1a1a2e"; input.style.border = "1px solid #415a77";
    input.style.borderRadius = "4px"; input.style.color = "#e0e0e0"; input.style.fontSize = "12px";
    input.onblur = () => onChange(input.value);
    wrap.appendChild(input);
    return wrap;
  }}

  async function batchUpdate(batchName, fields) {{
    await fetch("/batches/update", {{method:"POST", headers:{{"Content-Type":"application/json"}},
      body: JSON.stringify({{batch_name: batchName, ...fields}})}});
    batchSetStatus(`Saved changes to '${{batchName}}'.`);
  }}

  function batchSetStatus(msg) {{ document.getElementById("batch-status").textContent = msg; }}

  window.batchesInit();
}})();
</script>
"""
