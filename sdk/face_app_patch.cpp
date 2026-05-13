// face_app_patch.cpp
//
// Drop-in replacement of FaceApp::update_face_vertex_buffer that consumes a
// 502-point array (468 MediaPipe + 17 middle + 17 hairline).
//
// HOW TO INTEGRATE:
//
//   1. Replace the OBJ file path:
//        LoadOBJ("face_picture_3dmax.obj", obj_vertices, obj_indices);
//      ->
//        LoadOBJ("face_ext.obj", obj_vertices, obj_indices);
//
//   2. Replace the call to HardCodeData::Get().indexMap[i] with
//      head3d::kIndexMap502[i] (from ExtensionConstants.h), OR extend
//      hardcode_data.h's array to 502 entries with the contents from
//      ExtensionConstants.h.
//
//   3. Replace the body of update_face_vertex_buffer with the version below.
//
//   4. Update callers of update_face_vertex_buffer to pass 502 points.
//      Typical call site (today):
//          faceApp->update_face_vertex_buffer(pos468, 468);
//      becomes:
//          head3d::ExtensionPoints ext;
//          ext.LoadFromJson("data/<image>.json");
//          faceApp->update_face_vertex_buffer(ext.positions.data(), 502);
//
//   5. If the upstream still produces 468 MediaPipe points only (e.g. before
//      the Python pipeline runs), call update_face_vertex_buffer_partial()
//      below to update just the MP slice; the extension verts will keep
//      their canonical defaults from face_ext.obj.

#include "ExtensionConstants.h"
// The real integration points live in the SDK; the includes below are
// placeholders showing the dependencies your real FaceApp file already has.
//
// #include "FaceApp.h"
// #include <mutex>

// ---------------------------------------------------------------------------
// Replacement body for FaceApp::update_face_vertex_buffer
// ---------------------------------------------------------------------------
//
// void FaceApp::update_face_vertex_buffer(float* pos, int pointCount)
// {
//     if (!isInited()) return;
//     if (pointCount != head3d::kNumTotal) {
//         // Either pass all 502, or call update_face_vertex_buffer_partial.
//         return;
//     }
//     std::lock_guard<std::mutex> lock(mtx_point);
//     last_update_time = getCurrentTimeMillis();
//
//     for (int i = 0; i < (int)obj_vertices.size(); ++i)
//     {
//         const int mp_or_ext = head3d::kIndexMap502[i];
//         const int face_index = obj_vertices_map[mp_or_ext]; // identity for ext
//
//         const float x = pos[face_index * 3 + 0];
//         const float y = pos[face_index * 3 + 1];
//         const float z = pos[face_index * 3 + 2];
//
//         obj_vertices[i].pos[0] = x;
//         obj_vertices[i].pos[1] = y;
//         obj_vertices[i].pos[2] = z;
//     }
//     calculateVertexNormals(obj_vertices, obj_indices);
//     uploadVertexData();
// }

// ---------------------------------------------------------------------------
// Optional: partial update if upstream still has only 468 MP points.
// Keeps the extension verts at whatever was loaded from face_ext.obj.
// ---------------------------------------------------------------------------
//
// void FaceApp::update_face_vertex_buffer_partial(float* mp468, int pointCount)
// {
//     if (!isInited()) return;
//     if (pointCount != head3d::kNumMP) return;
//     std::lock_guard<std::mutex> lock(mtx_point);
//     last_update_time = getCurrentTimeMillis();
//
//     for (int i = 0; i < (int)obj_vertices.size(); ++i)
//     {
//         const int mapped = head3d::kIndexMap502[i];
//         if (mapped >= head3d::kNumMP) continue; // skip extension verts
//         const int face_index = obj_vertices_map[mapped];
//         obj_vertices[i].pos[0] = mp468[face_index * 3 + 0];
//         obj_vertices[i].pos[1] = mp468[face_index * 3 + 1];
//         obj_vertices[i].pos[2] = mp468[face_index * 3 + 2];
//     }
//     calculateVertexNormals(obj_vertices, obj_indices);
//     uploadVertexData();
// }
