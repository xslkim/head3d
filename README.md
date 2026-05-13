# head3d — MediaPipe Face Mesh with hairline extension

Pipeline that takes a face photo and produces a 502-point 3D mesh extending
**from the MediaPipe forehead boundary up to the real hairline**, so the
existing Vulkan texture-overlay renderer can paint effects in the hairline
area (decorative lines, painted-down hairline, etc.).

The base 468 vertices are unchanged; we **append** 17 middle-row + 17 hairline
vertices using a fixed topology, so the existing UV-driven shader pipeline
keeps working without modification.

See [PLAN_hairline.md](PLAN_hairline.md) for the design rationale.

## Project layout

```
head3d/
├── PLAN_hairline.md         # design doc
├── README.md                # this file
├── requirements.txt         # pip deps
├── face.obj                 # original MediaPipe canonical mesh (468 v)
├── face_ext.obj             # generated extended mesh (502 v, 916 tris)
├── python/
│   ├── constants.py             # MP_TOP_ANCHORS, UV layout, parser classes
│   ├── obj_io.py                # minimal OBJ reader/writer
│   ├── face_landmarks.py        # MediaPipe FaceMesh wrapper
│   ├── face_parsing.py          # HuggingFace SegFormer wrapper
│   ├── hairline_2d.py           # ray-cast hairline detection
│   ├── lift_3d.py               # 2D → 3D lifting
│   ├── build_extended_obj.py    # generates face_ext.obj (one-time)
│   ├── extract_hairline.py      # main CLI: image → 502-point JSON
│   ├── visualize.py             # debug overlays
│   └── _index_map_data.py       # auto-extracted indexMap from the SDK
├── sdk/
│   ├── ExtensionConstants.h     # 502-entry indexMap + anchor list
│   ├── ExtensionLoader.h/.cpp   # JSON → float[502*3] for C++ side
│   ├── face_app_patch.cpp       # commented reference of the patched function
│   └── README.md                # C++ integration guide
└── data/
    └── (generated JSONs and debug PNGs land here)
```

## Quickstart

### 1. Install deps

```powershell
py -3 -m pip install -r requirements.txt
```

`mediapipe`, `torch`, `torchvision`, and `transformers` are large; first run
also downloads ~150 MB of `jonathandinu/face-parsing` weights into the HF
cache.

### 1a. Download the MediaPipe model

```powershell
mkdir models
curl -L -o models/face_landmarker.task `
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 2. (Re)generate `face_ext.obj`

Already in the repo. If you edit `python/constants.py` (e.g. `MP_TOP_ANCHORS`
or the UV strip), rebuild:

```powershell
py -3 python/build_extended_obj.py
```

### 3. Run the full pipeline on an image

```powershell
py -3 python/extract_hairline.py path/to/photo.jpg --out data/photo.json
```

Output `data/photo.json` schema:

```json
{
  "image":   {"width": 1024, "height": 1024, "path": "..."},
  "n_total": 502,
  "n_mp":    468,
  "n_extension": 34,
  "layout":  ["mp[0..468)", "middle[468..485)", "hairline[485..502)"],
  "points":  [[x_norm, y_norm, z_rel], ... 502 entries ...],
  "valid_hairline": [true, true, ..., false, ...]   // 17 entries
}
```

### 4. Visualize what the pipeline saw

```powershell
py -3 python/visualize.py all path/to/photo.jpg --out data/photo_all.png
```

Produces a 3-panel image: parsing overlay | anchor↔hairline rays | extended mesh wireframe.

Individual panels:

```powershell
py -3 python/visualize.py parse   photo.jpg --out data/parse.png
py -3 python/visualize.py anchors photo.jpg --out data/anchors.png
py -3 python/visualize.py mesh    photo.jpg --out data/mesh.png
```

For the final mesh overlay (cyan MP boundary, orange middle row, yellow hairline):

```powershell
py -3 python/show_result.py photo.jpg data/photo.json --out data/photo_result.png
```

A pure-MediaPipe anchor preview that doesn't need torch:

```powershell
py -3 python/preview_anchors.py photo.jpg --out data/anchors_preview.png
```

### 5. Feed the JSON to the C++ SDK

See [sdk/README.md](sdk/README.md). One-line summary:

```cpp
head3d::ExtensionPoints ext;
ext.LoadFromJson("data/photo.json");
faceApp->update_face_vertex_buffer(ext.positions.data(), 502);
```

## Tuning knobs

All in `python/constants.py`:

| Constant | Purpose | When to tune |
|----------|---------|--------------|
| `MP_TOP_ANCHORS` | The 17 MediaPipe indices that define the upper boundary | If the rays in `visualize.py anchors` skip past actual hairline or come from a wrong spot |
| `curve_offset_z(i)` | Forehead backward curvature (deepest at center) | If the mesh looks too flat or too curved in side view |
| `bulge_z(i)` | Middle row inward bulge | Same |
| `UV_STRIP_U_MIN/MAX`, `UV_MIDDLE_V`, `UV_HAIRLINE_V` | Where the extension UVs live in the atlas | If your texture content collides with the strip |

After tuning, rerun `python/build_extended_obj.py` and reload `face_ext.obj`
in the SDK.

## Status by phase

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | Single image, frontal face | ✅ Python pipeline done; SDK files ready to drop in |
| **2** | Real-time (video / camera) | Not started — see `PLAN_hairline.md` §5 |
| **3** | Side-view faces | Not started — see `PLAN_hairline.md` §6 |

## What changed vs. original (`face.obj` → `face_ext.obj`)

```
vertices:  468 → 502   (+34)
texcoords: 468 → 502   (+34)
normals:   468 → 502   (+34)
triangles: 852 → 916   (+64)
```

The first 468 of everything is byte-identical to the original (modulo the
file's whitespace formatting). Existing UV bakes still work as-is.

## Known limitations

- **Bald / hat / hairline outside frame**: hairline points fall back to a fixed
  geometric extrapolation (`hairline_2d.py:sample_hairline:fallback_extrapolation`).
- **Heavy bangs**: detected "hairline" is the bottom of the bangs, which is
  the visually correct boundary for AR effects but not the anatomical hairline.
- **Side view**: only weakly handled in Phase 1. The occluded side will produce
  hairline samples drawn from the (possibly unreliable) opposite side of the
  parser output. Phase 3 plans to mask these out.
