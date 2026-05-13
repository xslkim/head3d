// ExtensionLoader.h
//
// Helpers for reading a 502-point position array from a JSON file produced
// by python/extract_hairline.py and for assembling a contiguous float buffer
// suitable for FaceApp::update_face_vertex_buffer.
//
// The intended call sequence:
//   ExtensionPoints ext;
//   ext.LoadFromJson("data/myimage.json");
//   faceApp->update_face_vertex_buffer(ext.positions.data(), 502);

#ifndef HEAD3D_EXTENSION_LOADER_H
#define HEAD3D_EXTENSION_LOADER_H

#include <array>
#include <string>
#include <vector>

#include "ExtensionConstants.h"

namespace head3d {

struct ExtensionPoints {
    // Flat (x, y, z) for 502 vertices.
    std::vector<float> positions;        // size = 502 * 3
    std::array<bool, kNumAnchors> valid_hairline{};
    int image_width  = 0;
    int image_height = 0;

    // Load from the JSON schema produced by extract_hairline.py.
    // Returns true on success. Uses nlohmann::json (already a dep in the SDK).
    bool LoadFromJson(const std::string& json_path);

    // Helper: zero-init positions to canonical defaults derived from face_ext.obj.
    // Useful as a fallback if no per-image data is available.
    void Reset();
};

} // namespace head3d

#endif // HEAD3D_EXTENSION_LOADER_H
