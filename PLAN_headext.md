# 方案：MediaPipe Mesh 横向 + 上向扩展（额头 + 侧脸外圈）

> **目标**：在已固定的 v1-hairline (502 顶点) 之上，把面部 mesh 横向扩展到太阳穴 / 颧骨外侧 / 耳前一线，最终得到一条完整的 "ribbon"（环形扩展带），覆盖额头上沿 + 侧脸外圈，让 3D 贴图能落在脸颊外侧、太阳穴、整个额头上，用作美妆 / 装饰贴图 / 发型边界 / AR 等。
>
> **范围**：只做额头 + 侧脸外圈（颧骨外侧、太阳穴、耳前），**不做**下颌外侧 / 耳廓 / 头顶后脑勺 / 头发内部。
>
> **里程碑**：v2-headext

---

## 0. 关键决策（已和你对齐过）

| 维度 | 选择 | 影响 |
|------|------|------|
| 覆盖范围 | 额头 + 侧脸外圈 (lateral) | 不动下颌/耳廓/后脑勺 |
| 应用场景 | 美妆 + 贴图 + 发型 + AR（混合） | 拓扑要规整、UV 要预留多类用途 |
| 3D 深度 | 椭球解析模型估算 Z | 小角度旋转贴图可用；不引入 FLAME/HRN |
| 输入形态 | 先做照片，拓扑预留视频 | 拓扑/编号必须稳定，不能跟着图像内容变 |
| 与 v1-hairline 关系 | 复用现有 17 hairline + 17 middle | 不改动 face_ext.obj 已有顶点编号 |

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Python 离线处理                                                  │
│                                                                  │
│   照片 ──┬─→ MediaPipe ─────────→ 468 个 3D 点                    │
│          │                                                       │
│          └─→ Face Parsing ──→ (face_skin ∪ hair) silhouette mask  │
│                                                                  │
│   现有 v1-hairline 输出（已固定）：                                 │
│     • 17 middle row [468..484]                                   │
│     • 17 hairline   [485..501]                                   │
│                                                                  │
│   新增 (v2-headext)：                                             │
│     • L 个 lateral_mid  [502..501+L]    — 侧脸中间行              │
│     • L 个 lateral_out  [502+L..501+2L] — 侧脸外圈 (silhouette)   │
│     • 椭球拟合 head_ellipsoid(a, b, c, R, t)                      │
│       └─ 给所有新顶点解算 Z                                       │
│                                                                  │
│   输出：face_ext_v2.obj  (468 + 34 + 2L 顶点 + 新增三角面)         │
│                                                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼ 通过 OBJ + JSON 传给 SDK
┌──────────────────────────────────────────────────────────────────┐
│  C++ / Vulkan                                                    │
│                                                                  │
│   FaceApp::update_face_vertex_buffer(pos_full[(502+2L)*3])       │
│     ├─ [0..467]       MediaPipe                                  │
│     ├─ [468..501]     v1-hairline 17+17                          │
│     ├─ [502..501+L]   lateral_mid (新)                           │
│     └─ [502+L..501+2L] lateral_out (新)                          │
└──────────────────────────────────────────────────────────────────┘
```

> 渲染管线**继续零侵入**，只动：① OBJ 文件升级为 `face_ext_v2.obj`；② 顶点 buffer 长度变成 `502 + 2L`；③ `sdk/ExtensionConstants.h` 同步更新。

---

## 2. 选哪些 MP 边界点作为 lateral 锚点

MediaPipe 468 点的左右脸轮廓里，**额头下沿 → 颧骨外侧 → 耳前下方**的链如下（右脸由左往右数为镜像）：

| 角色 | 左脸 MP idx | 右脸 MP idx | 说明 |
|------|-------------|-------------|------|
| 太阳穴上 (∩ hairline 端点) | 127 | 356 | v1-hairline 的端点锚点 1 / 17 |
| 太阳穴下 | 234 | 454 | v1-hairline 锚点 2 / 16 |
| 颧骨外侧上 | 93  | 323 | 颧弓上沿外侧 |
| 颧骨外侧中 | 132 | 361 | 耳前上方 |
| 耳前 / 腮上 | 58  | 288 | lateral ribbon 下终点 |

→ **每侧选 5 个 lateral 锚点**：`[127, 234, 93, 132, 58]` 和镜像 `[356, 454, 323, 361, 288]`，**L = 5**。

总扩展点数 = `2 × L × 2 = 20`（左 5 mid + 左 5 out + 右 5 mid + 右 5 out）。

新拓扑总顶点 = `502 + 20 = 522`。

> L=5 是一个保守的起点。如果做完看见侧脸 ribbon 太"瘦"，可以在相邻锚点之间线性插入中间种子点（沿用 `lateral_extend_dense` 的思路），用同一个 `intermediates` 参数 web 端实时调试，最终把 L 固定到合适值（预计 5–8）。

---

## 3. lateral 外圈点的 2D 求解

对每个 lateral 锚点 (e.g. MP 234)：

1. **方向**：沿"水平向外"方向，即 face-up 的垂直方向 (`perp = (-up.y, up.x)`)，左脸为负，右脸为正。
2. **射线**：从 MP 锚点位置出发，沿 `perp` 方向行进，命中 `(face_skin ∪ hair)` silhouette 外缘第一个**非掩码**像素的前一像素 → 即外圈点 `lateral_out[i]`。
3. **中间行**：`lateral_mid[i] = lerp(mp_anchor, lateral_out[i], 0.5)`，等距插在锚点和外圈之间。

可选用 `lateral_extend_dense` 已有的 `density_run_length`、`max_walk_ratio`（这里反过来：要走到 silhouette 边缘，不是停在 hair）、`use_adjacent` 等参数复用，web 端可调。

> **边界连续性**：左/右两端 lateral 链与现有 hairline 链顶端共用一对锚点（MP 127 / 356）。在 `face_ext_v2.obj` 里，hairline 端点和 lateral 端点的对应 mid / out 顶点共享 → 不会出现接缝。

---

## 4. Z 估算：椭球解析模型

给 lateral_mid 和 lateral_out 估 Z：不引入新模型，用 MP 468 点拟合一个椭球，给新点解析求 Z。

### 4.1 拟合椭球

对 MP 468 点 (`x_i, y_i, z_i`)，拟合一般二次曲面：

```
A x² + B y² + C z² + D xy + E xz + F yz
   + G x + H y + I z + J = 0
```

最小二乘解 10 维系数 (固定一个分量为 1 防尺度退化)，得到一个二次曲面 `Q(x, y, z) = 0`。

> 实际人头形状更接近 prolate ellipsoid（长轴前后向），468 点足够定形。如果数值不稳定，退化方案：直接用 PCA 估三个半轴 (a, b, c) + 中心 (cx, cy, cz) + 旋转 R，约束为标准椭球。

### 4.2 给新点求 Z

给定 `(x, y)`，把 `z` 当未知数代入 `Q(x, y, z) = 0`，解一元二次方程：
- 两个根分别对应椭球前后表面
- 取**靠近 MP 平均 Z** 的那个（前半球，与 MediaPipe Z 同侧）

### 4.3 兜底

如果 `(x, y)` 落在椭球水平投影外侧（数学上无实根），从最近 MP 锚点继承 Z + 一个小的法向偏移（沿椭球外法线方向往外推 ε 个单位），让贴图依然能跟上头部 yaw/pitch 小角度变化。

### 4.4 验证

在 web 服务里增加一个调试视图：把 522 个顶点投影到 2D，叠加在原图上，每个点画一个色块（按 Z 着色，红=近、蓝=远）。直观看是否合理。

---

## 5. 三角化和 UV

### 5.1 三角化（每侧 4 个 quad → 8 个三角面）

侧脸 ribbon 拓扑（左脸，右脸镜像）：

```
                          ┌─ hairline 链顶端 (485)
                          ▼
   lateral_out[0] ────┐
        │              \
   lateral_mid[0] ──── MP 127
        │              /
   lateral_out[1] ────┐
        │              \
   lateral_mid[1] ──── MP 234
        │              /
   ...
        │
   lateral_out[4] ────┐
        │              \
   lateral_mid[4] ──── MP 58
```

每对相邻 lateral 锚点 (i, i+1)，共 4 对 → 4 个 quad → 8 个三角面：

```
quad_i:  MP[i] ─ MP[i+1]
            │     │
         mid[i] ─ mid[i+1]
            │     │
         out[i] ─ out[i+1]
```

每 quad 拆 2 个三角面，共 **8 个 triangle / 侧**，**16 triangle / 总**。

外圈最末端 (lateral_out[0] 与现有 hairline_out[1]) 自然衔接（共享 MP 127 / 356）。

### 5.2 UV — 与现有贴图（`imgs/texture0.png`）的兼容性

实测 (`python python/uv_template.py face_ext.obj --underlay imgs/texture0.png --out imgs/uv_template_overlay.png`)：

| 区域 | OBJ V_raw 范围 | 512 图像 y 范围 | 含义 | 现有贴图占用 |
|------|----------------|-----------------|------|--------------|
| MP 468 主脸 | 0.0005 ~ 0.7724 | y = 116 ~ 511 | 完整脸部 UV，从下到上：chin → 鼻嘴 → 眼睛 → 额头 | `texture0.png` 内容在 y=229~460（眼鼻嘴一带），完全在这里 |
| v1 middle row 17 | 0.020 | y = 501 (1 行) | "UV 停车场"窄带 | 透明 |
| v1 hairline 17 | 0.005 | y = 509 (1 行) | 同上 | 透明 |
| **图像顶部空闲区** | (无顶点) | **y = 0 ~ 115（116 行）** | **完全空闲** | 透明 |

**结论 1：现有 `imgs/texture0.png` 直接套到 502 顶点（face_ext.obj）上和老 face.obj 渲染结果完全等价**——因为扩展顶点的 UV 落在 y=501/509 那两条贴图本来就透明的窄带上，sample 出来全透明，等于不绘制。

**结论 2：v2-headext 新增的 lateral_mid + lateral_out 顶点 UV 应该放进 y=0~115 这块空闲大区域**，而不是再去挤 y=501/509 那条已经只剩几像素的窄带。

v2 UV 设计：

- lateral 上半窄带：`V_raw ∈ [0.77, 0.78]` → 图像 y = [113, 117]（紧贴 MP 主脸 UV 顶端，但不与之重叠）
- lateral 下半窄带：`V_raw ∈ [0.79, 0.80]` → 图像 y = [102, 107]（mid 行）
- lateral 外圈窄带：`V_raw ∈ [0.81, 0.82]` → 图像 y = [92, 97]（out 行）
- U 方向：左 5 个锚点的 mid/out 顶点占 `U ∈ [0.05, 0.45]`，右 5 个占 `U ∈ [0.55, 0.95]`，中间留 `U=[0.45, 0.55]` 空白避免数值冲突

这样：

- 旧 512x512 贴图（内容只在 y=229~460）继续无修改可用，渲染结果与 face.obj 时代一致。
- 想给 lateral 区域加效果，只要在 y=0~120 区域上对应的 U 槽里画进去；可以用 `python/uv_template.py` 输出的 `imgs/uv_template.png` 当画稿底图。
- 如果未来效果纹理升到 1024×1024 / 2048×2048，UV 仍是同一套 [0,1] 归一化坐标，无需改 OBJ 或 SDK。

**渲染兼容性测试（阶段 C 验收项）**：

用 `texture0.png` 渲染 face.obj 和 face_ext_v2.obj，对中央脸部区域做 pixel diff，应当像素级一致（或≤1 LSB 差异）。lateral 和 hairline 区域因没有可贴像素应为透明 / 背景色。

---

## 6. 视频时序稳定性

只读图像无法保证 silhouette 检测帧间稳定。先做两件事：

1. **2D EMA**：`out_new = α · out_curr + (1-α) · out_prev`，α≈0.4。lateral_mid 自动跟着稳定。
2. **椭球 EMA**：椭球参数也做 EMA（α≈0.2），避免大幅抖动给 Z 抹出"波动"。

这两步在 v2 里以参数形式暴露但默认关闭（照片用不到，视频用），实现成本低。

---

## 7. Web 服务变化

`/headext` 命名空间，复用现有 `/hairline` 的 stem 缓存机制：

| 路径 | 作用 |
|------|------|
| `GET /headext` | 上传 + 调参主页 |
| `POST /headext/analyze` | 上传 + parse + landmark + 默认参数渲染 |
| `POST /headext/api/render` | 用 JSON params 重渲染 |
| `GET /headext/outputs/<filename>` | 拿渲染结果 |

调参 slider（暂列）：

| 参数 | 默认 | 作用 |
|------|------|------|
| `lateral_intermediates` | 0 | 锚点间是否加密 (与 hairline 一致) |
| `lateral_max_walk_ratio` | 0.15 | 外圈射线走距上限 (一般 15% 图宽，足够走到 silhouette) |
| `use_adjacent_silhouette` | ON | 走 face_adjacent 还是任意 hair |
| `ellipsoid_axes_clip` | 0.2 | 椭球外推时夹紧到椭球 0.2*半轴范围 |
| `temporal_alpha` | 1.0 | 时序 EMA (1.0 = 不平滑, 视频可调) |

可视化：

- 522 顶点全部画出 (按 [468/484/501/521] 分组配色)
- 16 个新 triangle 用线框叠在原图上
- Z 着色面板：每个顶点画成 (x,y) 圆点，颜色编码 Z

---

## 8. C++ 侧改动

| 文件 | 改动 |
|------|------|
| `face_ext_v2.obj` | 新文件，522 顶点 + 868 + 16 = 884 triangle |
| `sdk/ExtensionConstants.h` | `N_TOTAL = 522`, `LATERAL_MID_START`, `LATERAL_OUT_START` 等常量 |
| `sdk/FaceApp.cpp` | `update_face_vertex_buffer` 接受 522 个点 (向后兼容：长度判断) |
| `sdk/kIndexMap*` | 跟随 522 重新生成 |

> 兼容性策略：在 SDK 里增加 `N_TOTAL_VERSION` 字段；老 caller 传 502 仍然能跑，新功能用 522。

---

## 9. 分阶段实施

**阶段 A — 离线纯 Python 验证 (1 天)**

- A1. 实现 `sample_lateral_outer(landmarks, parse_map, ...) -> (10, 2)` 求 10 个外圈点 (左 5 右 5)
- A2. 实现 `fit_head_ellipsoid(landmarks_3d) -> Quadric` 椭球拟合
- A3. 实现 `lift_lateral_to_3d(out_xy, ellipsoid) -> (10, 3)` Z 解算
- A4. CLI 跑通：`python -m python.extract_headext <photo> --out data/photo_v2.json`，dump 522 顶点
- A5. 调试视图：`python python/visualize_headext.py photo --out data/photo_v2_overlay.png` 三联图 (silhouette, 522 点投影, Z 着色)

**阶段 B — Web 服务集成 (0.5 天)**

- B1. `/headext` 命名空间，复用 `HairlineWebAnalyzer` 缓存
- B2. slider 面板 + 实时调参
- B3. 与 `/hairline` 共存，两个页面分别工作

**阶段 C — OBJ 拓扑生成 + 文档 (0.5 天)**

- C1. 写一个 `tools/gen_face_ext_v2.py`，吃 522 个标准顶点 (从一张参考图算出) 加上人工 triangulation 配置 → 输出 `face_ext_v2.obj`
- C2. 校验：`python python/check_ext_obj.py face_ext_v2.obj`
- C3. 更新 README，新增 "扩展头部 mesh" 章节

**阶段 D — C++ 侧 (要你确认时机)**

- D1. `ExtensionConstants.h` 升级
- D2. `update_face_vertex_buffer` 长度判断
- D3. 离线生成的 522 顶点 JSON → C++ 加载测试 → 渲染验证

**阶段 E — 时序稳定 (可选，做视频时再加)**

- E1. EMA 平滑
- E2. silhouette 帧间一致性检测 (`lost_frame_threshold`)

**Tag**: 阶段 A + B + C 完成后打 `v2-headext`。D / E 视进度独立打。

---

## 10. 风险与备选

| 风险 | 后果 | 备选 |
|------|------|------|
| silhouette 在头发遮挡严重时丢边 | lateral_out 跑到头发内部 | 加密锚点 + 失效检测 + 椭球外推兜底 |
| 椭球拟合在极端头型上失真 | Z 估算偏 | 退化到 prolate ellipsoid + PCA |
| UV 左右窄条与 v1 hairline 顶部窄条冲突 | 贴图穿帮 | v3 统一重排 UV (现在先不动) |
| 美妆精度要求超过 ribbon 分辨率 | 美妆做不到细节 | 美妆走 468 点内的精细 UV，ribbon 只负责"外圈" |
| 头部 yaw 大角度 (>30°) | 椭球估 Z 失真，贴图错位 | 引入 FLAME (v3，需要时再做) |

---

## 11. 开发前需要你最后确认

- [ ] **L = 5** 起步，最后定 5–8（web 端调出来再固化）— OK?
- [ ] **椭球解析模型**而不是 FLAME — OK?
- [ ] **v2 UV 占图像顶部 116px 空闲大区** (而不是再挤窄带)；现有 `imgs/texture0.png` 直接兼容 face_ext_v2.obj，无需修改 — OK?
- [ ] **C++ 侧改动 (阶段 D) 等 Python 全部 ready 后再统一上**，不打散提交 — OK?
- [ ] **里程碑名字 v2-headext** — OK?

确认完我就按"阶段 A → B → C → 打 tag"执行。
