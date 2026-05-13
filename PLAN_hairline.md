# 方案：MediaPipe Mesh 扩展到发际线

> 目标：在不破坏现有 Vulkan 渲染管线的前提下，把面部 mesh 上沿从眉毛上方延伸到发际线之上，让你能用现有的"3D mesh + UV 贴图"方式画发际线特效（装饰线 / 下推填充等）。

---

## 0. 关键约束（从现有代码读出来的）

| 项 | 现状 | 扩展后约束 |
|----|------|----------|
| 顶点数 | 468 | 468 + M（M 待定，初始建议 17+17=34） |
| 拓扑 | 852 三角面（静态） | 852 + 新增三角面（静态） |
| UV | 每顶点固定（静态） | 每顶点固定（静态），新顶点的 UV 写死 |
| 坐标空间 | 图像归一化空间，X∈[0.17, 0.77], Y∈[0.28, 0.96], Z∈[-0.15, 0.29] | 同上，新顶点必须在同一坐标系 |
| 每帧 IO | `update_face_vertex_buffer(float* pos, int 468)` | 改为 `update_face_vertex_buffer(float* pos_full, int 468+M)` |
| 法线 | 每帧 `calculateVertexNormals` 重算 | 兼容（自动包含新顶点） |
| 着色器 | `texture.vert/frag`，UV 索引 8×4 子块图集 | **不动** |
| OBJ 文件 | `face.obj`（3dsMax 序 + indexMap 重映射） | 新文件 `face_ext.obj`，含扩展顶点和三角面 |

> 启示：**渲染管线零侵入**。整个扩展只动两个地方：① 替换 OBJ 文件；② `update_face_vertex_buffer` 多接受 M 个外部计算的 3D 点。

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  离线 / 预处理 (Python)                                       │
│                                                              │
│   照片  ──┬─→ MediaPipe  ────────────────→ 468 个 3D 点      │
│           │                                                  │
│           └─→ Face Parsing (BiSeNet/FaRL)                    │
│                     │                                        │
│                     ▼                                        │
│              hair / skin 像素掩码                              │
│                     │                                        │
│                     ▼                                        │
│              发际线 2D 曲线（取上边界）                          │
│                     │                                        │
│                     ▼                                        │
│        17 个锚点对齐采样 (与 MP_TOP_ANCHORS 一一对应)            │
│                     │                                        │
│                     ▼                                        │
│              2D → 3D 提升（Z 从相邻 MP 顶点继承 + 弧度补偿）       │
│                     │                                        │
│                     ▼                                        │
│              17 个发际线 3D 点 + 17 个中间行 3D 点（插值得到）     │
│                                                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼ 通过文件/socket/JNI 传给 SDK
┌─────────────────────────────────────────────────────────────┐
│  运行时 (C++ / Vulkan)                                        │
│                                                              │
│   FaceApp::update_face_vertex_buffer(pos_full[502*3])        │
│     │                                                        │
│     ├─ [0..467]   : MediaPipe 468 点（现有逻辑）               │
│     ├─ [468..484] : 中间行 17 点（Python 算出）                │
│     └─ [485..501] : 发际线 17 点（Python 算出）                │
│                                                              │
│   渲染 → 你已有的贴图特效管线（无需改动）                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 一次性定义的常量（最关键部分）

### 2.1 MP 上边界锚点列表 `MP_TOP_ANCHORS`

从 MediaPipe 468 点里挑出额头/太阳穴上沿的 17 个点，从左耳上方→额顶→右耳上方依次排列：

```cpp
// 候选起点（需要在 M1 阶段对照 MediaPipe canonical face 微调）
const int MP_TOP_ANCHORS[17] = {
    127, 234, 162,  21,  54, 103,  67, 109,  10,
    338, 297, 332, 284, 251, 389, 356, 454
};
```

> 这是 MediaPipe **原始 468 点的序号**，不是 OBJ 里的序号。OBJ 里的对应位置要通过 `indexMap` 反查（见 §4.3）。

### 2.2 扩展顶点 ID 规划

| ID 区间 | 含义 | 数量 |
|---------|------|------|
| 0 .. 467 | MediaPipe 原始 468 点 | 468 |
| 468 .. 484 | 中间行（额头中部） | 17 |
| 485 .. 501 | 发际线 | 17 |

合计 502 个顶点。

### 2.3 三角形索引扩展

每段相邻锚点之间形成两条三角形带（4 个三角形 × 16 段 = 64 个新三角面），加上左右两端的封口三角形（如果需要）。

总三角面：852 + 64 = 916（约值，最终以生成工具输出为准）。

### 2.4 扩展顶点的 UV（写死）

UV 空间是 4096×2048 纹理图集，按 8×4 = 32 个 512×512 子块切分。**给扩展区域单独分配一个"额头条带"UV 区域**，建议占用现有未使用的 UV 空白处。

具体 UV 坐标在 §3.3 的工具脚本里生成并写入新 OBJ。

---

## 3. Phase 1：单张图片，正脸（MVP）

### 3.1 环境准备

需要的 Python 依赖：
- `mediapipe` (已有)
- `torch` + face-parsing 权重（[zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)）
- `opencv-python`
- `numpy`

放到 `D:/face_sdk/python/` 下，与现有 `combin_png.py` 等工具同目录。

### 3.2 工具脚本 `extract_hairline.py`

输入：单张 RGB 图片
输出：JSON，包含 502 个 3D 点（468 原 + 34 扩展）

伪代码：

```python
def extract(image_path) -> dict:
    img = cv2.imread(image_path)
    H, W = img.shape[:2]

    # Step 1: MediaPipe
    landmarks_468 = mediapipe_face_mesh(img)  # shape (468, 3)
                                              # x,y 归一化到 [0,1]，z 是 MP 相对深度

    # Step 2: 分割
    parse_map = face_parser(img)              # (H, W) int, 类别索引
    hair_mask = (parse_map == HAIR_CLASS)
    skin_mask = (parse_map == SKIN_CLASS)

    # Step 3: 发际线 2D 曲线
    # = hair 区域和 skin 区域在脸部上方的分界线
    hairline_curve_2d = extract_upper_boundary(hair_mask, skin_mask)

    # Step 4: 锚点对齐采样
    hairline_pts_2d = []
    for mp_idx in MP_TOP_ANCHORS:                      # 17 次
        anchor_2d = landmarks_468[mp_idx, :2] * [W, H] # 反归一化
        up_dir = compute_face_up_vector(landmarks_468) # 简化：取 (0, -1)，正脸时够
        hit = ray_cast_to_curve(anchor_2d, up_dir, hairline_curve_2d)
        hairline_pts_2d.append(hit)

    # Step 5: 2D → 3D 提升
    hairline_pts_3d = []
    for i, p2d in enumerate(hairline_pts_2d):
        anchor_3d = landmarks_468[MP_TOP_ANCHORS[i]]
        z = anchor_3d[2] + CURVE_OFFSET_Z[i]            # 额头略向后弯
        x = p2d[0] / W                                  # 归一化回 [0,1]
        y = p2d[1] / H
        hairline_pts_3d.append([x, y, z])

    # Step 6: 中间行（线性插值 + 一点凸度）
    middle_pts_3d = []
    for i in range(17):
        a = landmarks_468[MP_TOP_ANCHORS[i]]
        b = hairline_pts_3d[i]
        mid = 0.5 * (a + np.array(b))
        mid[2] -= BULGE[i]                              # 额头隆起
        middle_pts_3d.append(mid.tolist())

    # Step 7: 拼接
    pos_full = np.concatenate([
        landmarks_468,
        np.array(middle_pts_3d),
        np.array(hairline_pts_3d),
    ])  # shape (502, 3)

    return {"points": pos_full.tolist()}
```

关键子函数：
- `extract_upper_boundary`：对 `hair_mask` 做形态学闭运算去毛刺，再求"每一列从下往上第一个 hair 像素的位置"得到曲线
- `ray_cast_to_curve`：从锚点向上找曲线第一个交点
- `CURVE_OFFSET_Z[17]`：预定义的 17 个数，中间深、两端浅；初值用 MediaPipe canonical face 的额头侧视轮廓量出来
- `BULGE[17]`：中间行的 Z 偏移，让额头有隆起感；初值 0.01 左右

### 3.3 生成新 OBJ：工具脚本 `build_extended_obj.py`

一次性运行，输出 `face_ext.obj`（502 顶点，~916 三角面）。

逻辑：

```python
def build():
    # 读现有 face.obj
    verts, uvs, normals, faces = parse_obj("D:/head3d/face.obj")

    # 把扩展部分追加到 OBJ
    # 占位顶点位置：用 MediaPipe canonical face 的均值脸算出来，作为静态默认
    # 实际运行时会被 update_face_vertex_buffer 覆盖
    canonical_extension_pos = compute_default_extension(verts)  # 34 个点

    # UV：分配在 UV 图的额头条带区域
    extension_uvs = make_extension_uv_strip()  # 34 个 UV

    # 法线：占位为 (0,0,1)，运行时重算
    extension_normals = [(0, 0, 1)] * 34

    # 三角面：连接 MP_TOP_ANCHORS → 中间行 → 发际线
    extension_faces = build_strip_topology()

    write_obj("D:/head3d/face_ext.obj",
              verts + canonical_extension_pos,
              uvs + extension_uvs,
              normals + extension_normals,
              faces + extension_faces)
```

> 注意：OBJ 顶点的 UV 必须固定。运行时只更新位置。

### 3.4 SDK 改动（C++ 侧）

最小化改动列表：

#### a) `hardcode_data.h`：扩展 indexMap

```cpp
// indexMap 现在长度 502。前 468 项保持不变，后 34 项是"恒等映射"
// （因为扩展顶点直接由 Python 算好，按相同索引顺序传进来）
const int indexMap[502] = {
    127, 34, 139, /* ... 原 468 项 ... */,
    // 扩展部分：恒等
    468, 469, 470, 471, 472, /* ... 到 501 ... */
};
```

#### b) 新增"扩展点"传输通道

```cpp
// FaceApp.h
void update_face_vertex_buffer(float* pos, int pointCount);
void update_extension_points(float* ext_pos, int extCount);  // 新增
// 内部缓存
float ext_points_[34 * 3];
bool  ext_points_valid_ = false;
```

#### c) `update_face_vertex_buffer` 改造

```cpp
void FaceApp::update_face_vertex_buffer(float* pos, int pointCount)
{
    if (!isInited()) return;
    std::lock_guard<std::mutex> lock(mtx_point);

    for (int i = 0; i < obj_vertices.size(); ++i)
    {
        int mapped_idx = HardCodeData::Get().indexMap[i];
        int face_index = obj_vertices_map[mapped_idx];

        float x, y, z;
        if (mapped_idx < 468) {
            // 原 MediaPipe 点
            x = pos[face_index * 3 + 0];
            y = pos[face_index * 3 + 1];
            z = pos[face_index * 3 + 2];
        } else if (ext_points_valid_) {
            // 扩展点（中间行 / 发际线）
            int ext_idx = mapped_idx - 468;
            x = ext_points_[ext_idx * 3 + 0];
            y = ext_points_[ext_idx * 3 + 1];
            z = ext_points_[ext_idx * 3 + 2];
        } else {
            // 回退：扩展点未提供 → 用 canonical 默认值（已在 OBJ 里）
            continue;
        }

        obj_vertices[i].pos[0] = x;
        obj_vertices[i].pos[1] = y;
        obj_vertices[i].pos[2] = z;
    }
    calculateVertexNormals(obj_vertices, obj_indices);
    uploadVertexData();
}
```

> 调用顺序：先 `update_extension_points`，再 `update_face_vertex_buffer`。也可以合并成一个接口，加 `int extCount` 参数。

#### d) OBJ 文件替换

把 `LoadOBJ("face_picture_3dmax.obj", ...)` 改为加载新生成的 `face_ext.obj`（注意是否走 3dsMax 顺序，需要复核）。

> ⚠️ 现有代码里 OBJ 名字是 `face_picture_3dmax.obj` 但 `D:/head3d/face.obj` 文件实际存在。确认实际部署用的是哪个版本的 OBJ，新 OBJ 要替换的是它。

### 3.5 UV 贴图改动

你需要在现有 UV 图集（4096×2048）的一个未用区域绘制：
- 额头中部条带（对应中间行 17 点）
- 发际线条带（对应发际线 17 点）

建议占用图集左下或右下一个 512×512 子块作为"额头扩展区"，由 push constants 切换到这个子块。

---

## 4. 验证里程碑

按这个顺序推进，每一步**都能直接看到中间结果**，方便调试：

| 阶段 | 工作量 | 产出 | 验证方式 |
|------|--------|------|----------|
| **M0** | 0.5d | 跑通 face-parsing.PyTorch | 用测试图，把 hair mask 红色叠在原图上保存 |
| **M1** | 0.5d | `extract_hairline.py` 完成 2D 部分 | 在原图上画出 17 个 2D 锚点 + 17 个 2D 发际线点 + 连接线 |
| **M2** | 0.5d | 2D → 3D 提升完成 | 把 502 个 3D 点投到图像平面，用 OpenCV 画线框看是否合理 |
| **M3** | 1d | `build_extended_obj.py` 生成 `face_ext.obj` | 用 Blender 打开，看拓扑、UV 是否合理 |
| **M4** | 1d | SDK 接入新 OBJ + 扩展点传输 | 渲染纯色填充，看额头扩展区是否完整显示 |
| **M5** | 0.5d | 画一条横跨发际线的装饰线贴图 | 验证 UV 对齐 |

总计约 4 人天到 MVP。

---

## 5. Phase 2：实时化（后续）

主要工作：
1. **分割模型替换**：face-parsing.PyTorch (~50ms CPU) → MediaPipe Image Segmenter 的 `multiclass_selfie_segmenter`（同生态、移动端实时、支持 hair 类）。或者用 BiSeNet 蒸馏后导出 TFLite 量化版。
2. **C++ / JNI 化**：把 `extract_hairline.py` 的核心逻辑移植到 C++，与现有 MediaPipe 调用合并到同一帧管线里。
3. **时序平滑**：发际线检测有抖动，对每个扩展顶点用 [One-Euro Filter](https://gery.casiez.net/1euro/) 平滑。
4. **失败回退**：分割失败/置信度低时，退回到几何外推（沿 MP_TOP_ANCHORS 法线方向按固定额高比例外推），避免 mesh 突变。

---

## 6. Phase 3：侧脸（更后续）

侧脸时一侧的锚点和发际线会被遮挡：

1. 用 MediaPipe 头部 yaw 角度判断遮挡侧
2. 遮挡侧扩展顶点改用"对称镜像"或几何外推（基于可见侧）
3. 在 vertex/fragment shader 加 per-vertex alpha 衰减，遮挡侧扩展三角面渐变到透明，避免穿帮

---

## 7. 边界情况处理

| 情况 | 检测 | 处理 |
|------|------|------|
| 秃头/发际线极高 | hair_mask 像素数 < 阈值 | 用固定比例外推（额高 = 1/3 脸高） |
| 刘海覆盖额头 | hairline_2d 比 MP 上沿低 | 视为正常 — 这就是用户视觉发际线，效果反而符合预期 |
| 戴帽子 | 帽子被错分为 hair，边缘僵直 | 用边缘平滑度检测帽檐（直线 vs 自然曲线），回退到外推 |
| 多人脸 | MediaPipe 返回多个人脸 | 沿用现有多人脸逻辑，每张脸独立跑分割 ROI |
| 闭眼/极端表情 | 不影响发际线 | 无需特殊处理 |

---

## 8. 待你决定的设计选择

在动工前确认：

1. **锚点数 N**：默认 17。更多 = 发际线更平滑但分割误差更显眼。
2. **是否要中间行**：默认要。不要的话三角形会很瘦长，光照偏。
3. **MP_TOP_ANCHORS 具体索引**：上面给的 17 个是候选起点，最好在 M1 阶段把这些点画在 canonical face 上肉眼确认覆盖发际线投影区域。
4. **扩展 UV 在图集的哪个位置**：取决于你现有贴图布局的空白处。
5. **Python 工具的部署形式**：
   - (a) 离线一次性算好，写到 JSON，C++ 端读 JSON 注入
   - (b) Python 进程常驻，通过 socket / 共享内存与 C++ 通信
   - (c) Phase 1 用 (a)，Phase 2 直接 C++ 重写
   
   建议 Phase 1 走 (a)，最简单。

---

## 9. 参考资源

- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) — BiSeNet 实现，19 类含 hair / skin
- [FaRL](https://github.com/FacePerceiver/FaRL) — 质量更高的备选
- [MediaPipe Image Segmenter](https://developers.google.com/mediapipe/solutions/vision/image_segmenter) — 实时阶段用
- [MediaPipe Face Mesh 468 点参考图](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png) — 锚点核对用
- [One Euro Filter](https://gery.casiez.net/1euro/) — 时序平滑

---

## 10. 风险与备注

- **OBJ 顺序的坑**：`face.obj` 是 3dsMax 导出，顶点顺序不是 MediaPipe canonical 顺序。SDK 里靠 `indexMap` 和 `obj_vertices_map` 两层映射处理。新 OBJ 生成脚本要保证扩展部分的顶点顺序与 SDK 端 `update_extension_points` 的输入顺序一致（建议直接用"恒等映射"避免再造一层）。
- **法线**：扩展顶点的法线由 `calculateVertexNormals` 自动算出，不用手动维护。
- **深度 Z 的尺度**：MediaPipe 的 Z 是相对单位，新顶点的 Z 必须与 MP 的 Z 在同一尺度。继承相邻 MP 顶点的 Z 是稳的做法。
- **不破坏现有 UV 贴图**：扩展部分用图集的"新区域"，原 468 顶点的 UV 完全不动，老贴图照常用。
