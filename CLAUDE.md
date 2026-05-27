# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**head3d** extends MediaPipe's 468-point face mesh into a **502-point mesh** with hairline and forehead detection:
- **468 original points** from MediaPipe FaceMesh (unchanged)
- **17 middle-row points** between MediaPipe boundary and hairline
- **17 hairline points** detected from hair/hat segmentation

The extension enables texture-based effects (decorative lines, hairline filling, etc.) on the forehead → crown region. Output is JSON (502 3D points) for SDK integration; also includes a web service for interactive tuning.

## Architecture at a Glance

### Pipeline Flow
```
Input Image
    ↓
[1] MediaPipe FaceLandmarker → 468 points + Z depth
    ↓
[2] HuggingFace SegFormer → semantic face segmentation (skin/hair/hat/background)
    ↓
[3] 2D Hairline Detection
    - Ray-cast from 17 fixed MediaPipe anchors upward along face-up direction
    - Stop at first hair/hat pixel (or fallback if not found)
    - Smooth with corner-aware binomial filter
    ↓
[4] 3D Lifting (Sagittal-arc model)
    - Hairline Z = anchor.Z + dy²/(2R), where dy=hairline.y - anchor.y, R=0.30×face_height
    - Middle row uses same formula with smaller dy → sits between anchor and hairline
    - Result: both rows curve backward naturally on head surface
    ↓
[5] Output: JSON or direct SDK integration (face_ext.obj ↔ C++ kIndexMap502)
```

### Key Design Decisions

**Sagittal-arc Model**: New points' Z values derive from a circular arc, ensuring they sit on the head surface (→z) rather than floating in front (←z). The `z = anchor.z + dy²/(2R)` formula is derived from the circular arc approximation; smaller R increases backward curve. This is critical—if Z ends up ≤ anchor.Z, the mesh is broken.

**Fixed 17 Anchors + Ray-casting**: Rather than fitting a global hairline curve, the algorithm casts rays from 17 fixed MediaPipe forehead points. This ensures point count and topology match `face_ext.obj` exactly. Valid-flag per point (`valid_hairline[]`) tracks whether ray hit real hair or fell back to geometric extrapolation.

**UV Layout with Anchor-Relative Offsets**: The 34 new vertices use `UV = anchor.UV + (DV_middle or DV_hairline)` in V, but inherit U directly from anchor. This keeps ribbon triangles vertical in UV space, preventing texture warping. If using uniform U (0.05..0.95) instead, the 5 hairline arcs get twisted into zig-zags.

**Corner-aware Smoothing**: The web service applies 1-2-1 binomial smoothing but skips points where adjacent neighbors form a sharp corner (cos(angle) < threshold). This preserves the upper-forehead/temporal transition while smoothing segmentation noise on the flat central forehead.

### Directory Structure

```
head3d/
├── python/                    # Python extraction pipeline
│   ├── constants.py           # Shared tuning constants (SYNC with C++ header!)
│   ├── face_landmarks.py      # MediaPipe FaceLandmarker wrapper
│   ├── face_parsing.py        # HuggingFace SegFormer semantic segmentation
│   ├── hairline_2d.py         # Ray-cast + smooth + validity detection
│   ├── lift_3d.py             # Sagittal-arc model (2D → 3D)
│   ├── build_extended_obj.py  # Generate face_ext.obj from face.obj
│   ├── extract_hairline.py    # Main CLI entry point
│   ├── web_service.py         # Flask server (/hairline + /preview routes)
│   ├── visualize.py           # Debug overlays (parse/anchors/mesh)
│   ├── preview_anchors.py     # Lightweight anchor preview (no torch needed)
│   ├── show_result.py         # Visualize final 502 points on input image
│   ├── uv_template.py         # Generate UV layout reference PNG
│   ├── obj_io.py              # OBJ file reader/writer
│   └── _index_map_data.py     # Extracted 468→OBJ vertex mapping from SDK
├── sdk/                       # C++ integration
│   ├── ExtensionConstants.h   # kIndexMap502, anchor indices, UV layout
│   ├── ExtensionLoader.h/.cpp # Read JSON → float[502*3] buffer
│   ├── face_app_patch.cpp     # Reference: how to patch update_face_vertex_buffer
│   └── README.md              # C++ integration checklist
├── face.obj                   # Original MediaPipe canonical mesh (468 verts)
├── face_ext.obj               # Extended mesh (502 verts) — regenerate if constants change
├── requirements.txt           # Python dependencies
├── start.sh                   # Web service launcher (bg/fg/status/logs)
└── PLAN_headext.md            # Historical design doc (reference only)
```

## Environment Setup

### Virtualenv & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies**:
- `mediapipe` — FaceLandmarker (models loaded from `models/face_landmarker.task`)
- `torch`, `torchvision`, `transformers` — SegFormer face parsing (≈150 MB weights auto-downloaded)
- `opencv-python`, `Pillow` — image I/O
- `Flask` — web service

### Download MediaPipe Model
```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Common Commands

### Single Image Processing
```bash
# Extract 502-point JSON from image
python python/extract_hairline.py path/to/photo.jpg --out data/photo.json

# With CUDA (if available)
python python/extract_hairline.py path/to/photo.jpg --out data/photo.json --device cuda
```

### Visualization & Debugging
```bash
# All-in-one debug overlay: parsing + anchors + mesh wireframe
python python/visualize.py all path/to/photo.jpg --out data/photo_all.png

# Individual visualizations
python python/visualize.py parse path/to/photo.jpg --out data/parse.png
python python/visualize.py anchors path/to/photo.jpg --out data/anchors.png
python python/visualize.py mesh path/to/photo.jpg --out data/mesh.png

# Preview anchors without loading SegFormer (fast, no torch needed)
python python/preview_anchors.py path/to/photo.jpg --out data/anchors_preview.png

# Show final 502 points overlaid on input
python python/show_result.py path/to/photo.jpg data/photo.json --out data/result.png
```

### Web Service
```bash
# Start in background (default port 18001)
./start.sh start

# Run in foreground (see logs in terminal)
./start.sh fg

# Check status
./start.sh status

# View logs
./start.sh logs

# Stop background service
./start.sh stop
```

**Access**: http://127.0.0.1:18001/hairline (main tuning UI), http://127.0.0.1:18001/preview (3D validation page)

### Mesh Regeneration (When Constants Change)
```bash
# Regenerate face_ext.obj after modifying constants.py
python python/build_extended_obj.py

# Regenerate UV layout reference PNG
python python/uv_template.py face_ext.obj --out imgs/uv_template.png
```

**When to regenerate**:
- If you change `MP_TOP_ANCHORS` (anchor positions)
- If you adjust `HEAD_ARC_RADIUS_FRAC` (curve radius)
- If you adjust `HAIRLINE_CROWN_LIFT_FRAC` (crown height lift)
- If you change UV offsets (`UV_MIDDLE_DV`, `UV_HAIRLINE_DV`)
- After regenerating, also update C++ `sdk/ExtensionConstants.h` to stay in sync

## Key Constants (python/constants.py)

These **MUST stay in sync** with C++ `sdk/ExtensionConstants.h`:

| Constant | Default | Purpose | Sync Required |
|----------|---------|---------|---------------|
| `MP_TOP_ANCHORS` | 17 indices | Forehead anchor points (start of ray-cast) | YES |
| `HEAD_ARC_RADIUS_FRAC` | 0.30 | Sagittal curve radius (fraction of face height) | YES |
| `HAIRLINE_CROWN_LIFT_FRAC` | 0.06 | Crown lift along face-up (fraction of face height) | YES |
| `UV_MIDDLE_DV` | 0.110 | Middle row V offset (anchor-relative) | YES |
| `UV_HAIRLINE_DV` | 0.220 | Hairline row V offset (anchor-relative) | YES |
| `fallback_extrapolation` | 0.18 | Ray fallback distance (if no hair found) | NO |

When you change any "Sync Required" constant:
1. Modify `python/constants.py`
2. Run `python python/build_extended_obj.py` → updates `face_ext.obj` geometry & UVs
3. Update `sdk/ExtensionConstants.h` with new values
4. If geometry changed, run `python python/uv_template.py` to refresh the UV reference PNG

## Testing & Validation

### Visual Inspection Workflow
1. **Parse Check**: Does `visualize.py parse` show hair properly segmented (hair ≠ skin)?
2. **Anchor Check**: Do `visualize.py anchors` rays start from correct forehead points and hit real hairlines?
3. **Mesh Check**: Does `visualize.py mesh` show a smooth extension band above the MP forehead boundary?
4. **3D Sanity Check** (on `/preview`): Is `⟨z(hairline) − z(anchor)⟩ > 0`? If ≤0, the Z model is broken.

### Web Service Iteration
1. Upload image on `/hairline` page
2. Adjust 7 sliders in real-time (intermediates, max_walk_ratio, etc.)
3. See rendered points/curves update in ~45 ms
4. Click `/preview` link to 3D-validate the 502 points with ortho overlay and texture checking

### Command-Line Validation
```bash
# Quick JSON check (502 points, valid_hairline flags, etc.)
python -c "import json; d=json.load(open('data/photo.json')); print(f'n_total={d[\"n_total\"]}, valid={sum(d[\"valid_hairline\"])}/{len(d[\"valid_hairline\"])}')"
```

## Integration with C++ SDK

The C++ side reads the JSON output and applies 502 points to the vertex buffer:

**Files to integrate**:
1. `sdk/ExtensionConstants.h` — kIndexMap502 (468→502), anchor indices, UV values
2. `sdk/ExtensionLoader.h` / `.cpp` — read JSON, fill float[502*3]
3. Update OBJ loader: load `face_ext.obj` instead of `face.obj`
4. Update vertex buffer loop: iterate 502 instead of 468

**Verify sync**:
- `kIndexMap502` in C++ must match order of `points[]` JSON array (468 MP, 17 middle, 17 hairline)
- `ExtensionConstants.h` constants (`HEAD_ARC_RADIUS_FRAC`, `HAIRLINE_CROWN_LIFT_FRAC`, UV offsets) must match `constants.py`
- OBJ vertex count (502) must match buffer size

See `sdk/README.md` for complete checklist.

## Troubleshooting

**Hairline position wrong**: Follow the order in README.md §排查建议:
1. Check `visualize.py parse` — is hair segmented correctly?
2. Check `visualize.py anchors` — are rays starting and direction right?
3. If anchors wrong → adjust `MP_TOP_ANCHORS`
4. If segmentation wrong → try different image/lighting or swap face parsing model
5. If 3D shape wrong → tune `HEAD_ARC_RADIUS_FRAC` or check `z(hairline) − z(anchor)` on `/preview`

**Import errors (mediapipe crash on WSL)**: The `web_service.py` uses a subprocess-based landmark backend by default to isolate MediaPipe's EGL issues. Use `--landmark-backend subprocess` (default) or set `HEAD3D_LANDMARK_BACKEND=subprocess`.

**Out of memory on SegFormer**: Switch device to CPU: `python extract_hairline.py photo.jpg --out data/photo.json --device cpu`

**OBJ mesh looks wrong**: Did you forget to regenerate after changing constants? Run `python python/build_extended_obj.py` and verify `face_ext.obj` changed.

## Historical & Design Context

- **PLAN_hairline.md**, **PLAN_headext.md** — Earlier design docs (reference only, not current truth)
- **Phase 1** (current): Single image, near-frontal faces, JSON output + web tuning
- **Phase 2** (future): Real-time video / camera pipeline
- **Phase 3** (future): Large yaw angles, occlusion-aware handling

## Web Service Architecture

### Routes
- `/hairline` — interactive tuning UI with 7 adjustment sliders
- `/hairline/analyze` — upload + run full pipeline (caches MediaPipe/SegFormer)
- `/hairline/api/render` — re-render with new slider params (uses cache, ~45 ms)
- `/preview` — 3D validation page (ortho overlay, texture upload, UV reference)
- `/health` — status check

### Caching
`HairlineWebAnalyzer` maintains an LRU cache (8 images) of `(rgb, parse_map, landmarks)`. Upload once, then slider adjustments are CPU-only (no model inference). Upload a different image to clear cache.

### Rendering Modes
- **v1-legacy** (single strategy): 17 fixed anchors, ray-cast, smooth
- **v1-hairline** (current): 17 + 16×N intermediates, lateral extension for outer points, corner-aware smooth, 7 tunable parameters

## Known Limitations & Assumptions

- **Sagittal-arc only**: Assumes circular arc in the sagittal plane. Asymmetry (yaw > ~30°) not fully handled.
- **Segmentation dependent**: Quality depends on HuggingFace SegFormer accuracy; no manual correction available.
- **Head-cropping**: If top of head cut off, rays can't reach real hairline → fallback to geometric extrapolation.
- **Bald / very high hairline**: Ray-cast may fail; fallback distance applies.
- **Only first face**: `FaceLandmarkerOptions(num_faces=1)`. Multi-face images only process the largest/first detected.
- **Thick bangs**: Treated as "visual hairline" (correct for texture overlay, not anatomically precise).
- **Z is relative, not real depth**: MediaPipe Z is relative depth + arc offset, not absolute camera-space depth.
