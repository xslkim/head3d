# SDK integration

These files are drop-in additions for a Vulkan-based face renderer that already
follows the project's existing pattern (`FaceApp::update_face_vertex_buffer` +
`indexMap[468]` + textured OBJ).

| File | What it is |
|------|------------|
| `ExtensionConstants.h` | Sizes, `MP_TOP_ANCHORS`, and the 502-entry `kIndexMap502` (replaces `indexMap[468]`). |
| `ExtensionLoader.h/.cpp` | Reads a `data/*.json` produced by `python/extract_hairline.py` and exposes a 502-float position buffer. |
| `face_app_patch.cpp` | Commented-out reference body of the patched `update_face_vertex_buffer`. Copy into your real `FaceApp.cpp`. |

## Integration checklist

1. Copy `ExtensionConstants.h`, `ExtensionLoader.h`, and `ExtensionLoader.cpp` into your SDK source tree (anywhere your build system already picks up `.h/.cpp`).
2. Either replace your `indexMap[468]` literal with `head3d::kIndexMap502` (from `ExtensionConstants.h`) or extend your own `hardcode_data.h` array to 502 entries with the same contents.
3. Change the OBJ load path from `face_picture_3dmax.obj` (468 verts) to `face_ext.obj` (502 verts). The file lives at the project root.
4. Replace the body of `FaceApp::update_face_vertex_buffer` with the version commented inside `face_app_patch.cpp`. The change is small: loop runs to 502 and reads from the extended index map.
5. At call sites, swap the 468-point feed for the 502-point feed produced by `ExtensionPoints::LoadFromJson(...)`.

## Vertex buffer growth

| Buffer | Before | After |
|--------|--------|-------|
| `obj_vertices` | 468 entries | 502 entries (+7%) |
| `obj_indices`  | 852 × 3 = 2556 | 916 × 3 = 2748 (+7.5%) |

Everything in `createVertexBuffer`, `uploadVertexData`, and the descriptor set
setup is sized from `obj_vertices.size()` / `obj_indices.size()` already, so
no buffer-size constants need to change.

## Shader / UV impact

Zero shader changes. The new 34 vertices have static UVs assigned to a
band near the top of the UV atlas (see `python/constants.py` →
`UV_MIDDLE_V`, `UV_HAIRLINE_V`). Tune those constants and re-run
`python/build_extended_obj.py` if the band collides with existing texture content.

## Runtime data flow

```
input image
   │
   │  py -3 python/extract_hairline.py image.jpg --out data/image.json
   ▼
data/image.json   (502 normalized 3D points)
   │
   │  ExtensionPoints::LoadFromJson("data/image.json")
   ▼
ExtensionPoints.positions  (float[502*3])
   │
   │  faceApp->update_face_vertex_buffer(ext.positions.data(), 502)
   ▼
Vulkan vertex buffer → existing texture pipeline
```

For real-time, Phase 2 of `PLAN_hairline.md` replaces the JSON hop with a
C++ port of the Python pipeline and feeds positions directly each frame.
