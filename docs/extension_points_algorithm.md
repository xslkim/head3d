# 扩展点算法详细文档

## 1. 概述

head3d 项目将 MediaPipe FaceMesh 的 468 个人脸关键点扩展为 **502 个点**，在额头上方增加了两圈共 34 个扩展点：

- **第一圈（Middle Row）**：17 个点，位于 MediaPipe 边界和发际线之间（索引 468–484）
- **第二圈（Hairline Row）**：17 个点，位于发际线位置（索引 485–501）

这 34 个扩展点用于在额头 → 头顶区域渲染贴图效果（装饰线条、发际线填充等）。

```
        ·  ·  ·  ·  ·  ·  ·   ← Hairline Row (485–501)
        ·  ·  ·  ·  ·  ·  ·   ← Middle Row  (468–484)
     ──────────────────────    ← MediaPipe 额头边界（17个锚点）
     |                    |
     |   MediaPipe 468点   |
     |                    |
     ──────────────────────
```

---

## 2. 总体流程

```
输入照片
    ↓
[Step 1] MediaPipe FaceLandmarker → 468 个 3D 关键点 (x_norm, y_norm, z_relative)
    ↓
[Step 2] HuggingFace SegFormer → 语义分割图 (皮肤/头发/帽子/背景等 19 类)
    ↓
[Step 3] 2D 发际线检测 → 17 个 2D 发际线点 (x_norm, y_norm)
    ↓
[Step 4] 3D 抬升 (Sagittal-arc model) → 17 个 hairline 3D 点
    ↓
[Step 5] Middle Row 生成 → 17 个 middle 3D 点
    ↓
[Step 6] 组装 502 点 → 输出 JSON / 渲染到 3D 模型
```

---

## 3. Step 1：MediaPipe 人脸关键点检测

使用 MediaPipe FaceLandmarker 检测 468 个人脸关键点，每个点包含 `(x_norm, y_norm, z_relative)`：

- `x_norm`, `y_norm`：归一化到 [0, 1] 的图像坐标
- `z_relative`：相对深度，鼻尖最小（最靠前），耳朵 / 脸侧较大（向后）

代码位置：`python/face_landmarks.py`

### 17 个锚点（Anchors）

扩展算法的起始点是 MediaPipe 额头上边界的 **17 个锚点**，定义在 `python/constants.py`：

```python
MP_TOP_ANCHORS = [
    127, 234, 162,  21,  54, 103,  67, 109,  10,
    338, 297, 332, 284, 251, 389, 356, 454,
]
```

这 17 个点从正面看**从左到右**排列，分布在额头上缘、太阳穴、眉梢外侧等位置。每个锚点都会向上方"发射"一条射线来寻找对应的发际线点。

---

## 4. Step 2：语义分割（Face Parsing）

使用 HuggingFace 的 SegFormer 模型 (`jonathandinu/face-parsing`) 对输入图片进行像素级语义分割，输出一个与图片同尺寸的标签图 `parse_map`，每个像素对应一个类别：

| 类别 ID | 含义 |
|---------|------|
| 0 | 背景 |
| 1 | 皮肤 |
| 13 | 头发 |
| 14 | 帽子 |
| ... | 其他面部部位 |

代码位置：`python/face_parsing.py`

关键掩码构建（`python/hairline_2d.py`）：

- **`build_hair_mask(parse_map)`**：提取 hair (13) + hat (14) 像素
- **`build_skin_mask(parse_map)`**：提取面部皮肤相关像素（skin/nose/brows/eyes/ears/lips 等）
- **`build_face_adjacent_hair_mask(parse_map)`**：只保留与面部皮肤相邻的头发像素，过滤掉被错分为 hair 的远处背景

---

## 5. Step 3：2D 发际线检测

### 5.1 面部朝上方向（Face-up Vector）

从下巴（MP #152）指向额顶（MP #10）的单位向量，定义了"头部向上"的方向：

```python
def face_up_vector(landmarks_norm):
    chin = landmarks_norm[152, :2]
    top  = landmarks_norm[10,  :2]
    v = top - chin
    return v / np.linalg.norm(v)
```

这个方向在 2D 图像空间中表示人脸的"上"方，即使头部有轻微倾斜也能正确追踪。

### 5.2 基础策略：沿 face-up 射线投射

**核心算法** (`sample_hairline`, `sample_hairline_baseline`)：

```
对于每个锚点 i (共17个):
    1. 获取锚点在图像中的像素坐标 anchor_px
    2. 沿 face-up 方向逐像素步进（步长 1px）
    3. 遇到的第一个 hair/hat 像素即为 hairline 点
    4. 如果射线走出图像边界仍未命中 → 使用 fallback：
       hairline[i] = anchor + face_up × 0.18 (归一化距离)
```

代码实现 (`python/hairline_2d.py:ray_first_hair_hit`)：

```python
def ray_first_hair_hit(origin_xy, direction_xy, hair_mask, max_steps=800, step_px=1.0):
    pos = origin_xy.copy()
    for _ in range(max_steps):
        pos += direction_xy * step_px
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        if 越界: return None
        if hair_mask[iy, ix]: return pos
    return None
```

### 5.3 /preview 页面的策略：轮廓行走（Silhouette Walk）

在 `/preview` 页面中，使用了一种**不依赖头发检测**的纯几何策略，适用于秃头、戴帽子等所有头型：

```
对于每个锚点 i:
    1. 计算该锚点在额头边界曲线上的法线方向（指向头部外侧/上方）：
       - 用相邻两个锚点的连线的切线
       - 取两个垂直候选方向中，指向远离脸部中心那个
    2. 沿此法线方向逐像素行走
    3. 用 head_mask（皮肤 ∪ 头发）判断是否在头部区域内
    4. 记录最后一个 head_mask=True 的位置
    5. 连续 5 个非头部像素则停止（确认已到达头部边界）
    6. 行走上限 = GEOMETRIC_HAIRLINE_OFFSET_FRAC × face_height 对应的像素 + 50px
```

法线方向计算：

```python
# 边界曲线切线
if 首端点:     tangent = anchor[1] - anchor[0]
elif 末端点:   tangent = anchor[-1] - anchor[-2]
else:          tangent = anchor[i+1] - anchor[i-1]

# 取指向外侧的法线
n1 = (-tangent[1], tangent[0])
n2 = (tangent[1], -tangent[0])
# 选与 (锚点 - 脸部中心) 方向更一致的那个
walk_dir = n1 if dot(n1, to_anchor) > dot(n2, to_anchor) else n2
```

这确保中心锚点几乎直上，而太阳穴锚点会向侧上方行走，跟随头骨形状。

### 5.4 /hairline 页面的高级策略

`/hairline` 页面使用 `sample_hairline_lateral_extend_dense`，是最成熟的检测策略：

#### (a) 加密采样

在 17 个 MP 锚点之间线性插值，每对之间插入 `intermediates` 个中间种子点：

```
总点数 = 17 + 16 × intermediates
当 intermediates=1 时：33 个采样点
当 intermediates=2 时：49 个采样点
```

#### (b) 密度门限射线

可选地要求射线必须连续命中 `density_run_length` 个 hair 像素才算正式命中，过滤掉单像素分类噪声：

```python
def ray_first_dense_hit(origin, direction, mask, run_length, max_steps=2000):
    consecutive = 0
    for each step:
        if mask[pixel]:
            consecutive += 1
            if consecutive >= run_length: return 起始位置
        else:
            consecutive = 0
```

#### (c) 横向延伸

对最外侧的锚点（太阳穴附近），从射线命中位置沿**同一行向外水平行走**，推到可见头发的真实边缘：

```
只修改 X 坐标，Y 保持不变
最大行走距离 = max_walk_ratio × 图像宽度（默认 1.5%）
每步检查下一个像素是否仍是 hair，否则停止
```

### 5.5 平滑处理

#### 基础平滑 (1-2-1 Binomial Filter)

```python
new[i] = 0.25 * out[i-1] + 0.5 * out[i] + 0.25 * out[i+1]
```

迭代 2 次。只对 valid（射线命中了真实头发的）点做平滑。

#### 角点感知平滑 (Corner-aware)

在平滑前计算相邻三点的夹角：

```python
v1 = P[i] - P[i-1]
v2 = P[i+1] - P[i]
cos_angle = dot(v1, v2) / (|v1| × |v2|)

if cos_angle < corner_cos_threshold:  # 默认 0.6（约 53°）
    保持原始位置（不平滑，保留转角）
else:
    应用 1-2-1 平滑
```

这保留了 M 形或梯形发际线的"肩部"转角，只平滑平坦区域的分割噪声。

---

## 6. Step 4：3D 抬升（Sagittal-arc Model）

### 6.1 核心公式

代码位置：`python/lift_3d.py`

将 2D 发际线点抬升为 3D 点的关键在于 **Z 坐标的计算**。模型假设头部矢状面（从前到后）的截面是一段**圆弧**：

```
z_hairline = z_anchor + dy² / (2R)

其中：
    z_anchor = 对应 MP 锚点的 Z 深度
    dy = y_hairline - y_anchor（发际线点与锚点的 Y 差值）
    R = HEAD_ARC_RADIUS_FRAC × face_height = 0.30 × 人脸高度
```

**物理直觉**：从额头向上走到头顶，头部表面逐渐向后弯曲。`dy²/(2R)` 就是一段圆弧上，从切线出发走了 `dy` 的弧长后，偏离切线方向的距离（即沿 +z 向头后偏移的量）。

```
              ← 头部前方(-z)    头部后方(+z) →
                    |
                    |  ·  ← hairline（z = z_a + dy²/2R，向后偏移）
                    | /
                    |·    ← middle（z 偏移较小）
                    ·     ← anchor（z = z_a，MP 原始 Z）
                    |
                    |     ← 额头（face.obj 表面）
```

### 6.2 Crown Lift（发冠抬升）

检测到的发际线是**皮肤与头发的分界线**，但贴图带需要覆盖到**头冠**位置。因此在 Z 计算之前，先沿 face-up 方向额外抬升 2D 坐标：

```python
lift = face_up_vector × (HAIRLINE_CROWN_LIFT_FRAC × face_height)
# 默认 HAIRLINE_CROWN_LIFT_FRAC = 0.06，即向上偏移 6% 的人脸高度

x_lifted = x_hairline + lift[0]
y_lifted = y_hairline + lift[1]
# 然后用 (x_lifted, y_lifted) 计算 Z
```

### 6.3 参数 R 的影响

| R (HEAD_ARC_RADIUS_FRAC) | 效果 |
|--------------------------|------|
| 0.15 (小) | 曲率大，hairline Z 向后偏移明显，紧弧 |
| 0.30 (默认) | 中等曲率，适合大多数人头 |
| 0.50 (大) | 曲率小，hairline Z 向后偏移较少，缓弧 |

### 6.4 数学推导

在矢状面上，以锚点为原点，Y 轴沿头部 "上" 方向，Z 轴向头后方向，头部表面近似为半径 R 的圆弧。

对于圆弧上位于角度 θ 处的点：
```
y = R sin(θ) ≈ Rθ      (小角近似)
z = R(1 - cos(θ)) ≈ Rθ²/2 = y²/(2R)
```

因此 `dz = dy²/(2R)` 本质上是**圆弧的抛物线近似**，在角度 < 46°（θ < 0.8 rad）时非常精确。

---

## 7. Step 5：Middle Row 生成

代码位置：`python/lift_3d.py:build_middle_row`

Middle row 位于锚点和 hairline 之间。早期方案用 XY 线性插值 + 独立 Z 弧公式，但由于 `dz ∝ dy²`，middle 只获得 25% 的 hairline Z 偏移，产生可见的凹陷。

**当前方案**：沿圆弧的**角度参数化**进行插值。

```
对每个锚点-hairline 对：
    θ_hairline = |dy_hairline| / R     # hairline 在圆弧上的角度
    θ_middle = bias × θ_hairline       # middle 的角度（默认 bias=0.5，即取半角）

    X_middle = (1 - bias) × X_anchor + bias × X_hairline  # X 线性插值

    # Y 和 Z 通过圆弧三角函数计算
    dy_middle = sign × R × sin(θ_middle)
    dz_middle = R × (1 - cos(θ_middle))

    Y_middle = Y_anchor + dy_middle
    Z_middle = Z_anchor + dz_middle
```

这确保 middle 点精确落在圆弧表面上，从 anchor 到 middle 到 hairline 的过渡平滑无凹陷。

---

## 8. /preview 页面的 10 种 3D 算法对比

`/preview` 页面同时计算 10 种不同的 3D 抬升算法，供可视化对比：

| 编号 | 名称 | Hairline Z | Middle 方法 |
|------|------|-----------|------------|
| A | 抛物线 R=0.30 mid=0.5 (原始) | `dy²/(2×0.30×fh)` | XY 插值 + 独立弧 Z |
| B | 抛物线 R=0.30 mid=0.75 | 同上 | XY 插值 (bias=0.75) + 独立弧 Z |
| C | XYZ 线性插值 50% | `dy²/(2×0.30×fh)` | XYZ 全线性插值 |
| D | 圆弧角度等分 θ/2 | `dy²/(2×0.30×fh)` | 角度参数化圆弧 (与 lift_3d.build_middle_row 相同) |
| E | 球面拟合 | 最小二乘球面投影 | 球面投影 |
| F | 切线 Hermite 插值 | `dy²/(2×0.30×fh)` | Hermite 样条 (用面部网格切线) |
| G | 三次贝塞尔曲线 | `dy²/(2×0.30×fh)` | Cubic Bezier (下巴→锚点切线) |
| H | 余弦缓动插值 | `dy²/(2×0.30×fh)` | Cosine easing XYZ 插值 |
| I | 缓弧 R=0.50 + Z线性 | `dy²/(2×0.50×fh)` | XYZ 全线性插值 |
| J | 紧弧 R=0.15 + Z线性 | `dy²/(2×0.15×fh)` | XYZ 全线性插值 |

默认显示 **C (XYZ 线性插值 50%)**。

---

## 9. 502 点组装与坐标顺序

### 9.1 组装

```python
def assemble_full(landmarks_norm, middle_3d, hairline_3d):
    # [0..468)  = MediaPipe 原始 468 点
    # [468..485) = Middle Row (17 点)
    # [485..502) = Hairline Row (17 点)
    return np.concatenate([landmarks_norm, middle_3d, hairline_3d])
```

### 9.2 坐标系转换

MediaPipe 输出的点序（landmark index 顺序）与 face.obj/face_ext.obj 中的顶点序不同。需要通过 `INDEX_MAP_468` 转换：

```python
# INDEX_MAP_468[obj_vertex_index] = mp_landmark_index
# 前 468 个需要重新排序，后 34 个扩展点直接复制
for obj_idx, mp_idx in enumerate(INDEX_MAP_468):
    pts_obj[obj_idx] = pts_mp[mp_idx]
pts_obj[468:] = pts_mp[468:]
```

---

## 10. 3D 网格生成（face_ext.obj）

代码位置：`python/build_extended_obj.py`

### 10.1 顶点位置

在 OBJ 的规范空间中（不依赖具体照片），扩展点位置使用同样的矢状弧模型：

```python
hairline_lift = (0.12 + crown_lift_frac) × face_h   # 0.12 是基础偏移
middle_lift = 0.5 × hairline_lift

y_hairline = y_anchor - hairline_lift    # OBJ 中 +Y 朝下，-Y 是"上"
z_hairline = z_anchor + (y_h - y_a)² / (2R)
```

### 10.2 三角面构建

每两个相邻锚点之间构建 4 个三角形（2 个 quad）：

```
Hairline[i] ─── Hairline[i+1]
     |  \            |
     |    \          |
Middle[i] ─── Middle[i+1]       ← 上层 band (2 个三角形)
     |  \            |
     |    \          |
Anchor[i] ─── Anchor[i+1]      ← 下层 band (2 个三角形)
```

16 对相邻锚点 × 4 个三角形 = **64 个扩展三角面**，加上原始 852 面 = **916 面总计**。

### 10.3 UV 坐标

每个扩展点的 UV 基于对应锚点的 UV **平行偏移**：

```python
U_middle   = U_anchor           # U 直接继承（保持 ribbon 列垂直）
V_middle   = V_anchor + 0.110   # V 上移固定 Δ

U_hairline = U_anchor
V_hairline = V_anchor + 0.220
```

**为什么用相对偏移而非固定值？**

face.obj 中 17 个锚点的 V 值是非均匀的弧形分布（中央 V≈0.77，太阳穴 V≈0.47，跨度达 0.30）。如果 middle/hairline 用固定 V 值，每列 ribbon 的 V 跨度差异巨大（中央 0.05，两端 0.35，差 7 倍），导致贴图上的弧线被不均匀拉伸。使用 anchor V + 固定 Δ 后，每列的 V 跨度恒定，贴图 5 条弧线在网格上粗细均匀。

---

## 11. 关键常量汇总

| 常量 | 默认值 | 作用 |
|------|--------|------|
| `N_MP` | 468 | MediaPipe 点数 |
| `N_ANCHORS` | 17 | 额头锚点数 |
| `N_TOTAL` | 502 | 总点数 (468 + 34) |
| `HEAD_ARC_RADIUS_FRAC` | 0.30 | 矢状弧半径（人脸高度的比例） |
| `HAIRLINE_CROWN_LIFT_FRAC` | 0.06 | 发冠抬升量（人脸高度的 6%） |
| `GEOMETRIC_HAIRLINE_OFFSET_FRAC` | 0.25 | /preview 纯几何偏移（无需头发检测） |
| `UV_MIDDLE_DV` | 0.110 | Middle 行 V 偏移量 |
| `UV_HAIRLINE_DV` | 0.220 | Hairline 行 V 偏移量 |
| `fallback_extrapolation` | 0.18 | 射线未命中时的回退距离 |

---

## 12. 完整算法流程图

```
┌──────────────────────────────────────────────────────────────┐
│                        输入照片                              │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
    ┌──────────────────┴──────────────────┐
    ↓                                     ↓
┌───────────────┐                 ┌───────────────┐
│  MediaPipe    │                 │   SegFormer   │
│  FaceLandmarker│                │  Face Parsing │
│  → 468 个 3D  │                │  → 像素级标签图│
│    关键点      │                │  (H×W)        │
└──────┬────────┘                 └──────┬────────┘
       │                                 │
       │  landmarks (468,3)              │  parse_map (H,W)
       │                                 │
       └──────────┬──────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   提取 17 个锚点坐标       │
     │   MP_TOP_ANCHORS           │
     │   计算 face_up_vector      │
     │   构建 hair_mask           │
     └────────────┬───────────────┘
                  ↓
     ┌────────────────────────────────────────────────┐
     │   对每个锚点沿 face-up 方向射线投射              │
     │                                                │
     │   for i in range(17):                          │
     │     从 anchor[i] 出发                           │
     │     逐像素沿 face-up 方向行走                    │
     │     命中 hair_mask → hairline_2d[i] = hit_pos  │
     │     未命中 → hairline_2d[i] = anchor + 0.18×up │
     └────────────┬───────────────────────────────────┘
                  ↓
     ┌────────────────────────────┐
     │   角点感知平滑              │
     │   1-2-1 binomial × 2 次   │
     │   cos < 0.6 时保留转角     │
     └────────────┬───────────────┘
                  ↓
     ┌────────────────────────────────────────────┐
     │   Crown Lift                               │
     │   hairline_2d += face_up × 0.06 × face_h  │
     └────────────┬───────────────────────────────┘
                  ↓
     ┌────────────────────────────────────────────────┐
     │   3D Z 计算 (Sagittal-arc)                     │
     │                                                │
     │   for i in range(17):                          │
     │     dy = y_hairline[i] - y_anchor[i]           │
     │     R = 0.30 × face_height                     │
     │     z_hairline[i] = z_anchor[i] + dy²/(2R)    │
     │                                                │
     │   → hairline_3d (17, 3)                        │
     └────────────┬───────────────────────────────────┘
                  ↓
     ┌────────────────────────────────────────────────┐
     │   Middle Row 生成 (圆弧角度参数化)               │
     │                                                │
     │   for i in range(17):                          │
     │     θ_h = |dy_hairline| / R                    │
     │     θ_m = 0.5 × θ_h                           │
     │     X_m = 0.5 × X_anchor + 0.5 × X_hairline   │
     │     Y_m = Y_anchor + sign × R × sin(θ_m)      │
     │     Z_m = Z_anchor + R × (1 - cos(θ_m))       │
     │                                                │
     │   → middle_3d (17, 3)                          │
     └────────────┬───────────────────────────────────┘
                  ↓
     ┌────────────────────────────────────────────────┐
     │   组装 502 点                                   │
     │   [landmarks(468) | middle(17) | hairline(17)] │
     │                                                │
     │   → points_full (502, 3)                       │
     └────────────┬───────────────────────────────────┘
                  ↓
     ┌────────────────────────────────────────────────┐
     │   用 INDEX_MAP_468 转换为 OBJ 顶点顺序          │
     │   覆盖 face_ext.obj 的 502 个顶点位置           │
     │   Three.js 渲染 3D 模型 + 照片贴图              │
     └────────────────────────────────────────────────┘
```

---

## 13. 验证要点

1. **Z 方向正确性**：`z(hairline) - z(anchor) > 0`，即 hairline 点必须在锚点后方（向头后方偏移）。如果 ≤ 0，说明 Z 模型计算错误。
2. **平滑度**：从 anchor → middle → hairline 的 Z 过渡应平滑无凹陷。
3. **UV 对齐**：ribbon 列在 UV 空间中应保持垂直（U 与锚点一致）。
4. **点数一致**：总点数必须恰好 502 = 468 + 17 + 17。
