// ExtensionLoader.cpp
//
// Implementation of head3d::ExtensionPoints::LoadFromJson.
// Depends on nlohmann::json (already a dep in the host SDK).

#include "ExtensionLoader.h"

#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace head3d {

void ExtensionPoints::Reset() {
    positions.assign(kNumTotal * 3, 0.0f);
    valid_hairline.fill(false);
    image_width = 0;
    image_height = 0;
}

bool ExtensionPoints::LoadFromJson(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        return false;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception&) {
        return false;
    }

    int n_total = j.value("n_total", 0);
    if (n_total != kNumTotal) {
        return false;
    }
    const auto& pts = j["points"];
    if (!pts.is_array() || (int)pts.size() != kNumTotal) {
        return false;
    }

    positions.resize(kNumTotal * 3);
    for (int i = 0; i < kNumTotal; ++i) {
        const auto& p = pts[i];
        if (!p.is_array() || p.size() != 3) {
            return false;
        }
        positions[i * 3 + 0] = p[0].get<float>();
        positions[i * 3 + 1] = p[1].get<float>();
        positions[i * 3 + 2] = p[2].get<float>();
    }

    valid_hairline.fill(false);
    if (j.contains("valid_hairline") && j["valid_hairline"].is_array()) {
        const auto& v = j["valid_hairline"];
        for (int i = 0; i < kNumAnchors && i < (int)v.size(); ++i) {
            valid_hairline[i] = v[i].get<bool>();
        }
    }

    if (j.contains("image")) {
        image_width  = j["image"].value("width",  0);
        image_height = j["image"].value("height", 0);
    }
    return true;
}

} // namespace head3d
