# head3d - 带发际线扩展的 MediaPipe 3D 人脸网格

这个项目把单张人脸照片转换成一个 **502 点 3D 人脸网格**。它保留 MediaPipe FaceMesh 原始的 468 个点不变，并在额头上沿额外追加 34 个点：

- 17 个“中间行”点：位于 MediaPipe 额头上沿和真实发际线之间。
- 17 个“发际线”点：贴近图片里识别到的头发边界。

这样做的目的，是让已有的 Vulkan 贴图渲染管线可以把特效画到额头和发际线区域，例如装饰线、下推发际线、发际线填充等。原来的 468 点、人脸拓扑和 UV 仍然可用；新增部分只是在网格顶部加一条额头扩展带。

设计背景见 [PLAN_hairline.md](PLAN_hairline.md)。C++ 接入说明见 [sdk/README.md](sdk/README.md)。

## 一句话流程

输入一张照片后，程序会：

1. 用 MediaPipe 找到人脸 468 个 3D landmark。
2. 用 HuggingFace SegFormer 做人脸语义分割，得到每个像素属于皮肤、头发、帽子、五官等哪一类。
3. 从 MediaPipe 额头上沿的 17 个锚点出发，沿“脸部向上”的方向逐像素找第一个 hair/hat 像素，这 17 个命中点就是 2D 发际线采样点。
4. 把 2D 发际线点补上 Z 值，变成 17 个 3D 发际线点。
5. 在 MediaPipe 上沿和发际线之间插值生成 17 个中间行点。
6. 拼成 `468 + 17 + 17 = 502` 个点，输出 JSON，供 SDK 更新顶点缓冲。

## 项目结构

```text
head3d/
├── PLAN_hairline.md         # 中文设计文档，记录方案、阶段、风险和后续计划
├── README.md                # 当前说明
├── requirements.txt         # Python 依赖
├── face.obj                 # 原始 MediaPipe canonical mesh，468 个顶点，852 个三角面
├── face_ext.obj             # 额头扩展 mesh，502 个顶点，916 个三角面
├── python/
│   ├── constants.py             # 关键常量：17 个锚点、顶点布局、矢状-arc 半径、UV、分割类别
│   ├── obj_io.py                # 简单 OBJ 读写器
│   ├── face_landmarks.py        # MediaPipe FaceLandmarker 封装，输出 468 个点
│   ├── face_parsing.py          # HuggingFace SegFormer 人脸分割封装
│   ├── hairline_2d.py           # 发际线 2D 识别：mask + 射线搜索 + 平滑 + 回退
│   ├── lift_3d.py               # 2D 发际线点提升到 3D，z 用矢状-arc 反解
│   ├── build_extended_obj.py    # 一次性生成 face_ext.obj
│   ├── extract_hairline.py      # 主入口：图片 -> 502 点 JSON
│   ├── visualize.py             # 调试可视化
│   ├── web_service.py           # 本地 web 服务 (/hairline + /preview)
│   ├── uv_template.py           # 输出 UV 布局参考图
│   └── _index_map_data.py       # 从 SDK 提取的 468 点 indexMap
├── sdk/
│   ├── ExtensionConstants.h     # C++ 侧尺寸、锚点、502 项 indexMap
│   ├── ExtensionLoader.h/.cpp   # 读取 Python 输出 JSON，转成 float[502*3]
│   ├── face_app_patch.cpp       # update_face_vertex_buffer 的参考改法
│   └── README.md                # C++ 接入清单
└── data/
    └── 运行后生成的 JSON 和调试 PNG 建议放这里
```

## 环境准备

建议先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以用：

```powershell
py -3 -m pip install -r requirements.txt
```

依赖里比较大的包有 `mediapipe`、`torch`、`torchvision`、`transformers`。第一次运行人脸分割时，还会自动下载约 150 MB 的 `jonathandinu/face-parsing` 权重到 HuggingFace 缓存。

还需要下载 MediaPipe FaceLandmarker 模型：

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Windows PowerShell：

```powershell
mkdir models
curl -L -o models/face_landmarker.task `
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

`python/face_landmarks.py` 默认会从 `models/face_landmarker.task` 加载这个文件。

## 快速运行

生成一张图片对应的 502 点 JSON：

```bash
python python/extract_hairline.py path/to/photo.jpg --out data/photo.json
```

如果有 CUDA，也可以指定设备给人脸分割模型：

```bash
python python/extract_hairline.py path/to/photo.jpg --out data/photo.json --device cuda
```

输出 JSON 结构如下：

```json
{
  "image": {
    "width": 1024,
    "height": 1024,
    "path": "path/to/photo.jpg"
  },
  "n_total": 502,
  "n_mp": 468,
  "n_extension": 34,
  "layout": [
    "mp[0..468)",
    "middle[468..485)",
    "hairline[485..502)"
  ],
  "points": [
    [0.501, 0.312, -0.041]
  ],
  "valid_hairline": [
    true
  ]
}
```

实际 `points` 有 502 项，顺序固定：

| 下标范围 | 含义 | 数量 | 来源 |
|----------|------|------|------|
| `0..467` | MediaPipe 原始人脸点 | 468 | `FaceLandmarker.detect()` |
| `468..484` | 额头中间行 | 17 | `build_middle_row()` |
| `485..501` | 发际线行 | 17 | `sample_hairline()` + `lift_hairline_to_3d()` |

`valid_hairline` 有 17 项。某一项为 `true` 表示对应锚点的射线真的命中了 hair/hat 像素；为 `false` 表示没有命中，使用了几何外推回退点。

## 发际线是怎样识别出来的

核心代码在 `python/hairline_2d.py`，入口是：

```python
hairline_2d, valid = sample_hairline(landmarks, parse_map)
hairline_2d = smooth_hairline(hairline_2d, valid)
```

这里的识别不是靠传统边缘检测，也不是在整张图上找一条连续曲线；当前实现更稳定、更可控：先用语义分割知道哪里是头发，再从 17 个固定额头锚点向上找头发。

### 1. 先得到 MediaPipe 468 点

`python/face_landmarks.py` 使用 MediaPipe Tasks Vision 的 `FaceLandmarker`：

- 输入 RGB 图片。
- `num_faces=1`，只取第一张脸。
- MediaPipe task 模型可能输出 478 点，其中后 10 个是虹膜点；本项目只保留前 468 点，匹配原始 SDK 的 `face.obj`。
- 每个点是 `[x, y, z]`：
  - `x`、`y` 是归一化图像坐标，范围大致在 `[0, 1]`。
  - `z` 是 MediaPipe 的相对深度，不是毫米，也不是相机真实深度。

### 2. 固定 17 个 MediaPipe 上沿锚点

`python/constants.py` 里定义了 17 个额头上沿锚点：

```python
MP_TOP_ANCHORS = [
    127, 234, 162, 21, 54, 103, 67, 109, 10,
    338, 297, 332, 284, 251, 389, 356, 454,
]
```

这些是 MediaPipe 原始 468 点的 landmark 下标，按正脸视角从左到右排列。每个锚点都会向上采样一个发际线点，所以最终得到 17 个发际线点。

这样做有两个好处：

- 发际线点数量固定，和扩展 mesh 的拓扑、UV、SDK buffer 一一对应。
- 每个发际线点都绑定到一个额头锚点，后续 2D -> 3D 时可以直接继承附近锚点的深度。

### 3. 再做人脸语义分割

`python/face_parsing.py` 使用 HuggingFace 模型：

```text
jonathandinu/face-parsing
```

它会输出一张和原图同尺寸的 `parse_map`，每个像素是一个类别编号。类别定义在 `python/constants.py`，其中和发际线最相关的是：

| 类别常量 | 编号 | 含义 |
|----------|------|------|
| `PARSE_SKIN` | 1 | 皮肤 |
| `PARSE_HAIR` | 13 | 头发 |
| `PARSE_HAT` | 14 | 帽子 |

当前实现把 `hair` 和 `hat` 都算作可命中的“头发区域”：

```python
hair_mask = np.isin(parse_map, [PARSE_HAIR, PARSE_HAT])
```

也就是说，射线只要遇到被分割模型判为头发或帽子的像素，就认为到达了视觉上的头部上边界。对 AR 贴图来说，这通常比“解剖学发际线”更实用，因为刘海、帽檐、头发表面才是用户实际看到的边界。

### 4. 计算“脸部向上”方向

图片里的人脸可能略微歪头，所以不能简单使用屏幕坐标的 `(0, -1)`。`face_up_vector()` 用两个 MediaPipe 点估计脸的局部向上方向：

- 下巴点：`152`
- 额头顶部附近点：`10`

计算方式：

```python
up = normalize(landmarks[10, :2] - landmarks[152, :2])
```

因为图像坐标里 `y` 向下增大，所以正脸时这个向量大致指向屏幕上方。后面每个锚点都会沿这个方向发出射线。

### 5. 从每个锚点向上射线搜索第一个头发像素

对每个 `MP_TOP_ANCHORS[i]`：

1. 取这个 MediaPipe 点的归一化 2D 坐标。
2. 乘以图片宽高，转成像素坐标。
3. 沿 `up` 方向每次前进 1 像素。
4. 检查当前位置是否落在 `hair_mask` 上。
5. 第一次命中 hair/hat 像素，就返回这个位置作为该锚点对应的 2D 发际线点。

伪代码：

```python
for mp_idx in MP_TOP_ANCHORS:
    pos = landmarks[mp_idx, :2] * [W, H]
    while pos still inside image:
        pos += up_px * 1.0
        if hair_mask[round(pos.y), round(pos.x)]:
            hit = pos
            break
```

这个设计有一个重要细节：射线不会因为进入背景就停止，只会在命中 hair/hat 时停止。这样做是为了处理太阳穴、耳侧这些横向锚点，因为这些点往上走时可能先离开脸部皮肤区域，再碰到头发。如果遇到背景就停止，侧边锚点很容易误判为脸/背景边界，而不是发际线。

### 6. 如果找不到头发像素，就几何外推

如果射线走出图片或走完最大步数仍未命中 hair/hat，`sample_hairline()` 会用固定距离回退：

```python
hairline = anchor + up * fallback_extrapolation
```

默认：

```python
fallback_extrapolation = 0.18
```

这里的 `0.18` 是归一化图像坐标里的距离。回退常见于：

- 秃头或发际线很高，分割不到头发。
- 头顶超出图片边界。
- 分割模型漏判头发。
- 帽子、强阴影、背景颜色造成分割异常。

对应位置的 `valid_hairline[i]` 会写成 `false`，方便调试和 SDK 侧做额外处理。

### 7. 对发际线采样点做轻量平滑

语义分割边缘会有锯齿和局部抖动，所以命中 17 个点后会调用：

```python
smooth_hairline(hairline_2d, valid, iterations=2)
```

它对曲线中间点做两轮 `1-2-1` binomial 平滑：

```python
new[i] = 0.25 * p[i - 1] + 0.5 * p[i] + 0.25 * p[i + 1]
```

只有当前点 `valid[i] == true` 时才会更新该点；无效点会保留回退结果。两端点不参与平滑，避免边界收缩。

### 8. 把 2D 发际线点提升到 3D —— 矢状-arc 模型

`python/lift_3d.py` 的 `lift_hairline_to_3d()` 把每个 2D 发际线点变成 3D 点：

```python
x = hairline_2d[i].x
y = hairline_2d[i].y
dy = y - landmarks[MP_TOP_ANCHORS[i]].y        # 总是 < 0 (hairline 在 anchor 上方)
R  = HEAD_ARC_RADIUS_FRAC × face_height        # 默认 0.30 × face_h
z  = landmarks[MP_TOP_ANCHORS[i]].z + dy² / (2 R)
```

也就是说：

- `x`、`y` 来自实际识别到的发际线位置。
- `z` 用一个 **矢状-arc (sagittal-arc) 模型** 反解, 把头骨在中线 (x ≈ x_anchor) 上的纵向剖面当成局部圆弧, 半径 R 取脸高的 30 %。
- `dy² / (2R)` 是圆弧的弦-高近似 `R − R·cos(θ)`, 永远 ≥ 0 → **z 永远 ≥ anchor.z** → 在 MediaPipe / face.obj 约定下 (`-z = 脸前, +z = 头后`), 新点永远朝头后方向偏, 严格落在头骨表面, 而不是浮在脸前。

> ⚠️ 历史教训: 之前 `curve_offset_z(i) = -0.04 × cos(...)` 把 hairline z 推到 **比额头 anchor 更负 (更靠近相机)**, 等于把所有新点丢到脸的前面飘着。`/preview` 页能看到 `⟨z(hairline) − z(MP anchor)⟩` 这个指标, 应大于 0 才贴合, 这是 bug 没复发的硬性 sanity check。

### 9. 生成中间行

如果直接把 MediaPipe 上沿连到发际线行，三角面会又长又扁，贴图和光照都不自然。所以项目在中间加一行 17 个点：

```python
x = 0.5 * anchor.x + 0.5 * hairline.x
y = 0.5 * anchor.y + 0.5 * hairline.y
dy = y - anchor.y
z = anchor.z + dy² / (2 R)             # 同一个 arc 模型, 自动小于 hairline 那一行
```

middle 行用 **同一个** 矢状-arc 模型: 因为 `dy_middle² < dy_hairline²`, middle 的 z 自动落在 anchor 和 hairline 之间, 形成一个连续向后弯的曲面带, 三段贴在头骨上。最终新增的 34 个点就是：

- `middle[0..16]`
- `hairline[0..16]`

### 10. 拼成 SDK 需要的 502 点

最后 `assemble_full()` 按固定顺序拼接：

```python
points_full = concatenate([
    landmarks_468,
    middle_3d,
    hairline_3d,
])
```

这个顺序必须和 `face_ext.obj`、`sdk/ExtensionConstants.h`、`kIndexMap502` 保持一致。否则 C++ 侧会把点写到错误的 OBJ 顶点上。

## Web 服务

本地 Web 服务把发际线检测包装成"上传 → 实时调参"的交互界面。启动：

```bash
python python/web_service.py
```

默认监听 `0.0.0.0:8000`。**所有发际线相关页面与接口都在 `/hairline` 命名空间下**（根路径 `/` 会 302 重定向到 `/hairline`），方便后续在 `/texture`、`/mesh3d` 等同级路径里加新功能，不互相干扰：

```text
http://127.0.0.1:8000/hairline
```

### 固定策略：`lateral_extend_dense` + 转角保护平滑

经过多轮策略对比 (`baseline` / `adjacent` / `arch` / `window` / `density` / `inverse` / `boundary_proj` / ...) 已经确认 `sample_hairline_lateral_extend_dense` + `smooth_hairline_corner_aware` 在中间额头、上方转角、侧鬓三段都最贴合视觉发际线。Web 服务在 `v1-hairline` 之后固定只跑这一个策略，但通过 7 个可调参数 (slider) 暴露内部配置，让用户在线选取最适合自己照片的取值。

算法管线：

1. **MediaPipe + face parsing**（上传时跑一次，缓存）。
2. **种子加密**：在原始 17 个 `MP_TOP_ANCHORS` 之间按 `intermediates` 线性插入 N 个中间种子（默认 1，共 33 个种子点）。
3. **射线探测**：每个种子沿 face-up 方向直上，命中首个 `hair` 像素（可选 `use_adjacent` 过滤背景误分为 hair 的像素，可选 `density_run_length` 要求连续 N 个 hair 才停）。
4. **横向延伸**：最外侧 `outer_per_side` 个点沿 hit 横行向外侧推到可见 hair 边缘（上限 `max_walk_ratio × W`）。
5. **转角保护平滑**：1-2-1 binomial 平滑 `smooth_iters` 次，但相邻三点夹角小于 `corner_cos_threshold` 时跳过该点不平滑，保住额头转角。

| 参数 | 范围 | 默认 | 作用 |
|------|------|------|------|
| `intermediates` | 0–4 | 1 | 中间插入点数。最终点数 = 17 + 16 × N，即 17 / 33 / 49 / 65 / 81 |
| `max_walk_ratio` (×1000) | 0–50 | 15 | 外侧锚点横向延伸幅度上限 (单位 0.1% 图宽)，0 关闭 |
| `outer_per_side` | -1..15 | -1=auto | 应用横向延伸的外侧点数，-1 = `(intermediates+1)×3` |
| `density_run_length` | 0–30 | 0 | 必须连续命中 N 个 hair 像素才算命中，0 = 一像素即停 |
| `smooth_iters` | 0–5 | 2 | corner-aware 平滑迭代次数，0 = 不平滑显示 raw hit |
| `corner_cos_threshold` (×100) | 0–100 | 60 | `cos(夹角) <` 阈值视为转角不平滑；越小越严，越大越多点保留 raw |
| `use_adjacent` | bool | ON | 仅保留邻接 face skin 的 hair 像素，过滤背景误分 |

### 页面与 HTTP API

| 路径 | 方法 | 作用 |
|------|------|------|
| `/` | GET | 302 → `/hairline` |
| `/hairline` | GET | 上传 + 调参主页 |
| `/hairline/analyze` | POST (multipart) | 上传图片，跑 parse + landmark + 一次默认参数渲染，返回完整调参页 |
| `/hairline/api/render` | POST (JSON) | 用最新参数重渲染（命中缓存，~45 ms/帧） |
| `/hairline/outputs/<filename>` | GET | 返回原图或渲染好的点图/曲线图 |
| `/health` | GET | 健康检查，返回 `{ok, backend, cached: [stem...]}` |

`/hairline/api/render` 请求体：

```json
{
  "stem": "ae020b...user_red_hairline",
  "params": {
    "intermediates": 2,
    "max_walk_ratio_x1000": 15,
    "outer_per_side": -1,
    "density_run_length": 4,
    "smooth_iters": 2,
    "corner_cos_x100": 60,
    "use_adjacent": true
  }
}
```

响应：

```json
{
  "points_filename": "ae020b..._dense_points.png",
  "curve_filename":  "ae020b..._dense_curve.png",
  "n_total": 49,
  "n_valid": 49,
  "elapsed_ms": 44
}
```

页面里左侧 slider 触发 200 ms debounce 的 `/hairline/api/render` 请求，并用 `renderSeq` 单调递增的方式抛弃过期响应，所以拖动滑块时只看最新一帧。

### 缓存

`HairlineWebAnalyzer` 维护一个 LRU 缓存 (容量 8 张) 存 `(rgb, parse_map, landmarks)`。**上传一次 → 此后所有调参渲染都不再跑 MediaPipe 或 SegFormer**，所以 slider 拖动是 CPU-only 流水（数十毫秒）。需要换图就重新上传。

### 命令行参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` | `0.0.0.0` | 监听地址，局域网外访问用 `0.0.0.0` |
| `--port` | `8000` | 端口 |
| `--device` | `$HEAD3D_DEVICE` | face parsing 推理设备 (`cuda` / `cpu` / 默认自动) |
| `--landmark-backend` | `subprocess` | 见下表 |
| `--threaded` | off | 开启 Flask 多线程请求（默认关闭以避免 MediaPipe 在某些平台的线程问题） |
| `--debug` | off | Flask debug 模式 |

`--landmark-backend` 取值：

| backend | 说明 |
|---------|------|
| `subprocess` (默认) | 每次分析都启动一个子进程跑 MediaPipe FaceLandmarker，子进程在 `import mediapipe` 之前注入 `LIBGL_ALWAYS_SOFTWARE=1` / `MESA_LOADER_DRIVER_OVERRIDE=llvmpipe` / `MEDIAPIPE_DISABLE_GPU=1` / `EGL_PLATFORM=surfaceless` 等环境变量绕开 WSL 下 EGL 段错误；即使 native 代码崩了也只会杀掉子进程，Web 服务保持运行 |
| `tasks` | 进程内 `mediapipe.tasks.vision.FaceLandmarker`（环境稳定时最快） |
| `solutions` | 进程内 `mediapipe.solutions.face_mesh`（仅旧版 mediapipe 有） |
| `parsing` | 不跑 MediaPipe，只用 face parsing 估计发际线（用作终极兜底，会丢失锚点） |

上传图片与渲染结果都写在 `data/web/`，命名是 `<uuid>_<原文件名>_dense_(points|curve).png`。这个目录在 `.gitignore` 里，不入版本控制。

## 可视化和调试

最推荐先运行总览图：

```bash
python python/visualize.py all path/to/photo.jpg --out data/photo_all.png
```

输出是一张三联图：

| 面板 | 内容 | 用来检查什么 |
|------|------|--------------|
| parsing overlay | 人脸分割结果叠到原图上 | 头发是否被分成 hair/hat，皮肤是否误分 |
| anchor rays | 17 个 MediaPipe 锚点、17 个发际线点和连线 | 射线是否从正确位置出发，是否命中真实发际线 |
| mesh wireframe | 扩展区域三角网格投影 | 中间行和发际线行是否形成合理额头扩展带 |

也可以分别输出：

```bash
python python/visualize.py parse   path/to/photo.jpg --out data/parse.png
python python/visualize.py anchors path/to/photo.jpg --out data/anchors.png
python python/visualize.py mesh    path/to/photo.jpg --out data/mesh.png
```

如果只想看 MediaPipe 17 个锚点，不想加载 torch/分割模型，可以运行：

```bash
python python/preview_anchors.py path/to/photo.jpg --out data/anchors_preview.png
```

如果已经有 `data/photo.json`，可以把最终扩展结果画回原图：

```bash
python python/show_result.py path/to/photo.jpg data/photo.json --out data/photo_result.png
```

## 生成和更新 `face_ext.obj`

仓库里已经包含生成好的 `face_ext.obj`。只有改了这些内容时才需要重新生成：

- `python/constants.py` 里的 `MP_TOP_ANCHORS`
- 扩展顶点的 UV 常量
- Z 弧度参数
- `face.obj`
- SDK 的原始 `indexMap`

重新生成：

```bash
python python/build_extended_obj.py
```

生成逻辑：

1. 读取原始 `face.obj`。
2. 用 `_index_map_data.py` 里的 `INDEX_MAP_468` 建立 MediaPipe landmark 下标到 OBJ 顶点下标的反查表。
3. 按 17 个 `MP_TOP_ANCHORS` 追加 17 个中间行顶点和 17 个发际线顶点。
4. 给新增顶点分配固定 UV。
5. 新增 64 个三角面，把 `anchor row -> middle row -> hairline row` 连成两条三角带。
6. 写出 `face_ext.obj`。

生成后的差异：

```text
vertices:  468 -> 502   (+34)
texcoords: 468 -> 502   (+34)
normals:   468 -> 502   (+34)
triangles: 852 -> 916   (+64)
```

原始 468 个点、UV、法线和三角面保留在前面；新增内容只追加在文件末尾。运行时 SDK 会更新 502 个顶点的位置，法线由 SDK 重新计算。

## UV 和贴图

新增 34 个顶点的 UV 由 `python/constants.py` 的 `extension_uv_for(row, anchor_u)` 给出：

```python
UV_MIDDLE_V    = 0.820   # raw OBJ V  → image y ≈  92  (品红弧线 / 中间行)
UV_HAIRLINE_V  = 0.940   # raw OBJ V  → image y ≈  31  (蓝弧线 / 发际线行)
# U: 直接继承对应 MP_TOP_ANCHORS[i] 的 U
```

含义：

- **U**: middle[i] 和 hairline[i] **直接复制** MP_TOP_ANCHORS[i] 这个 anchor 的 U。 anchor 17 个 U 是非均匀的 (`0.001..0.999`, 太阳穴附近 ≈ 0, 额头中央 ≈ 0.5)，ribbon 三角形列只有让 middle/hairline 跟着 anchor 的 U 才会**在 UV 空间里垂直**, 贴图 5 条横向弧线才不会被 ribbon 三角形扭成 Z 字形。 ❌ 如果用 0.05..0.95 均匀分布 U, 弧线会被斜拉。
- **V**: anchor 行 V_raw ∈ [0.40, 0.77], middle 行 V_raw=0.82 (品红弧线), hairline 行 V_raw=0.94 (蓝弧线) — 全 17 列 anchor.V < middle.V < hairline.V 严格单调递增 (这是硬约束, anchor 最大 V 是额头中央 MP 10 的 0.7724, 所以 UV_MIDDLE_V 必须 > 0.7724 否则 ribbon 三角形 V 反向, 会出现局部贴图翻转折叠)。 品红和蓝两条弧线分别原位采到 middle / hairline 行, 其余 3 条弧线 (橙/红/紫) 通过 V 方向插值自动展开在 anchor → middle → hairline 的过渡带上。
- OBJ 里写的是 raw V, Three.js 用默认 `texture.flipY=true` 时 V_raw=1 直接对应贴图图片顶部, 不需要再翻转。

> ⚠️ 历史教训: 早期 `UV_MIDDLE_V=0.02 / UV_HAIRLINE_V=0.005` 把 ribbon UV 放到了贴图**底部**空白区 (img y ≈ 502..510), 结果 34 个新点采样全是白色, 发际线带完全看不见。 而贴图实际内容 (5 条彩色弧线) 一直在顶部 V_raw ≈ 0.67..0.94。 修复后, 在 `/preview` 顶部 ortho overlay 上能直接看到 5 条弧线投影在额头到发际线区域。

如果改了 `texture0.png` 把图块挪到别的位置, 同步调 `UV_MIDDLE_V` / `UV_HAIRLINE_V` (两个 V 值不一定相邻, 但要保证 anchor.V_min < UV_MIDDLE_V < UV_HAIRLINE_V <= 1, 否则 V 方向插值会反向), 然后重生成：

```bash
python python/build_extended_obj.py
```

## /preview 3D 端到端验证页

`/preview` 用来肉眼确认 502 点的 3D 位置和 OBJ canonical mesh 是否一致, 是这一轮 "新加的点是否贴皮肤" 的回归检查面板。

* **左** = 原图 + 502 识别点 (MP / v1 middle / v1 hairline 三组色编)。下方 meta 行打出 `⟨ z(hairline) − z(MP anchor) ⟩` 这个指标 —— **必须 > 0** 才说明 hairline 在 anchor 的后方 (头骨向后弯), 贴皮肤; 一旦 ≤ 0 就说明矢状-arc 模型回退到了直接抄 anchor Z 的旧 bug。
* **右上** = ortho 正交投影 overlay。原图当底, 贴图 mesh 叠在上面 (与原图严格像素对齐), 用来确认贴图 UV 与活脸 mesh 在画面上对得上。可切贴图/线框/不透明度。
* **右下** = `face_ext.obj` canonical 模板, 可拖动旋转。用来确认 canonical mesh 本身没有 z 翻号 / UV 错位。

HTTP：

| 路径 | 方法 | 说明 |
|------|------|------|
| `/preview` | GET | 3D 验证页上传入口 |
| `/preview/analyze` | POST (multipart) | 上传 + 算 502 点 |
| `/preview/api/data/<stem>` | GET | 该 stem 的 502 点 (MP-order + OBJ-order) |
| `/preview/assets/face_ext.json` | GET | OBJ 的 indexed-geometry JSON, Three.js 直接灌进 `BufferGeometry` |
| `/preview/assets/{face_ext.obj,texture0.png}` | GET | 静态资源 |

`/hairline` 页右上角有跳转链接；两个页面共享 stem，上传一次即可来回调。

## C++ SDK 接入

Python 侧生成 JSON 后，C++ 侧读取并直接传 502 点：

```cpp
head3d::ExtensionPoints ext;
ext.LoadFromJson("data/photo.json");
faceApp->update_face_vertex_buffer(ext.positions.data(), 502);
```

接入清单：

1. 把 `sdk/ExtensionConstants.h`、`sdk/ExtensionLoader.h`、`sdk/ExtensionLoader.cpp` 放进 SDK 工程。
2. 用 `head3d::kIndexMap502` 替换原来的 `indexMap[468]`，或按它的内容把原数组扩展到 502 项。
3. 把 OBJ 加载路径改成 `face_ext.obj`。
4. 参考 `sdk/face_app_patch.cpp` 修改 `FaceApp::update_face_vertex_buffer`，让循环覆盖 502 个顶点。
5. 调用侧从 468 点输入改成 `ExtensionPoints::LoadFromJson()` 读出来的 502 点输入。

缓冲增长很小：

| 内容 | 原来 | 现在 |
|------|------|------|
| 顶点 | 468 | 502，增加约 7% |
| 索引 | `852 * 3 = 2556` | `916 * 3 = 2748`，增加约 7.5% |

shader 和贴图采样逻辑不需要改；新增顶点已经在 OBJ 里有 UV。

## 可调参数

主要都在 `python/constants.py`：

| 参数 | 作用 | 什么时候调 |
|------|------|------------|
| `MP_TOP_ANCHORS` | 17 个 MediaPipe 额头上沿锚点 | 射线起点不合理、漏掉太阳穴、整体发际线偏移 |
| `HEAD_ARC_RADIUS_FRAC` | 矢状-arc 半径占脸高的比例 (默认 0.30) | hairline/middle 在 3D 里贴脸太紧或太靠后, 取值越小向头后弯曲越快 |
| `UV_MIDDLE_V` | 中间行贴图 V 坐标 | 中间行采样到错误贴图位置 |
| `UV_HAIRLINE_V` | 发际线行贴图 V 坐标 | 发际线边缘采样到错误贴图位置 |
| `fallback_extrapolation` | 射线找不到头发时的外推距离 | 秃头、头顶出框、分割失败时回退点太高或太低 |

改了 mesh 相关常量后，需要重新生成 `face_ext.obj` 并同步 SDK 常量。

## 当前阶段

| 阶段 | 范围 | 状态 |
|------|------|------|
| Phase 1 | 单张图片、偏正脸 | Python 管线已完成，SDK 接入文件已准备 |
| Phase 2 | 视频/相机实时 | 未开始，计划把分割和发际线逻辑移到实时 C++/移动端管线 |
| Phase 3 | 大侧脸、遮挡侧处理 | 未开始，计划根据 yaw 判断遮挡侧并做透明或镜像回退 |

## 已知限制

- 秃头、发际线极高或头顶超出画面时，射线可能找不到 hair/hat 像素，会退回固定几何外推。
- 厚刘海会被当成“视觉发际线”。这对贴图特效通常是正确的，因为用户看到的是刘海下沿，不是真实头发生长边界。
- 帽子被纳入 `hair_mask`，所以帽檐可能被当成边界。如果想严格识别真实头发，需要把 `PARSE_HAT` 从 `build_hair_mask()` 里移除。
- 当前只取第一张脸，`FaceLandmarkerOptions(num_faces=1)`。
- 大侧脸还没有完整处理，遮挡侧的发际线可能来自不可靠的分割结果。
- Z 值不是真实相机深度，只是 MediaPipe 相对深度加人工弧度偏移；它的目标是让渲染效果自然，不是医学或测量级 3D 重建。

## 排查建议

如果发际线位置不对，按这个顺序看：

1. 先看 `visualize.py parse`：头发有没有被分割成 hair/hat。
2. 再看 `visualize.py anchors`：17 个 MediaPipe 锚点是否在额头上沿，射线方向是否正确。
3. 如果锚点错，调 `MP_TOP_ANCHORS`。
4. 如果分割错，换图、改善光照，或换/微调 face parsing 模型。
5. 如果 2D 点对但 3D 形状不好，调 `HEAD_ARC_RADIUS_FRAC` (越小新点越向头后弯), 或上 `/preview` 看 `⟨z(hairline) − z(MP anchor)⟩` 这个指标是否为正。
6. 如果 mesh 对但贴图错，调 UV 常量并重新生成 `face_ext.obj`。
