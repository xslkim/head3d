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
├── face_ext.obj             # v1：额头扩展 mesh，502 个顶点，916 个三角面
├── face_ext_v2.obj          # v2：v1 + 20 个 lateral 顶点 (522 v, 948 tri)
├── python/
│   ├── constants.py             # 关键常量：17 个锚点、顶点布局、Z 偏移、UV、分割类别
│   ├── obj_io.py                # 简单 OBJ 读写器
│   ├── face_landmarks.py        # MediaPipe FaceLandmarker 封装，输出 468 个点
│   ├── face_parsing.py          # HuggingFace SegFormer 人脸分割封装
│   ├── hairline_2d.py           # 发际线 2D 识别：mask + 射线搜索 + 平滑 + 回退
│   ├── lift_3d.py               # 2D 发际线点提升到 3D，并生成中间行
│   ├── build_extended_obj.py    # 一次性生成 face_ext.obj
│   ├── extract_hairline.py      # 主入口 (v1)：图片 -> 502 点 JSON
│   ├── extract_headext.py       # 主入口 (v2)：图片 -> 522 点 JSON (含 lateral)
│   ├── head_ellipsoid.py        # v2：468 MP 点 -> 解剖学头部椭球 + Z 解算
│   ├── visualize.py             # 调试可视化 (v1)
│   ├── visualize_headext.py     # 调试可视化 (v2)：silhouette / 522 点 / Z 三联图
│   ├── web_service.py           # 本地 web 服务 (/hairline + /headext)
│   ├── uv_template.py           # 输出 UV 布局参考图
│   └── _index_map_data.py       # 从 SDK 提取的 468 点 indexMap
├── tools/
│   └── gen_face_ext_v2.py       # 一次性生成 face_ext_v2.obj
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

### 8. 把 2D 发际线点提升到 3D

`python/lift_3d.py` 的 `lift_hairline_to_3d()` 会把每个 2D 发际线点变成 3D 点：

```python
x = hairline_2d[i].x
y = hairline_2d[i].y
z = landmarks[MP_TOP_ANCHORS[i]].z + curve_offset_z(i)
```

也就是说：

- `x`、`y` 来自实际识别到的发际线位置。
- `z` 继承对应 MediaPipe 额头锚点的深度。
- 再加一个预设的额头弧度偏移 `curve_offset_z(i)`。

`curve_offset_z(i)` 是对称余弦形状：中间最深，两侧逐渐变浅。当前最大幅度约为 `-0.04`，目的是让头顶/额头上方看起来略微向后弯，而不是一整条平板。

### 9. 生成中间行

如果直接把 MediaPipe 上沿连到发际线行，三角面会又长又扁，贴图和光照都不自然。所以项目在中间加一行 17 个点：

```python
middle = 0.5 * anchor + 0.5 * hairline
middle.z += bulge_z(i)
```

`bulge_z(i)` 也是对称余弦形状，当前最大幅度约为 `-0.015`，用来给额头区域一点弧度。最终新增的 34 个点就是：

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

新增 34 个顶点有自己的静态 UV，定义在 `python/constants.py`：

```python
UV_STRIP_U_MIN = 0.05
UV_STRIP_U_MAX = 0.95
UV_MIDDLE_V = 0.02
UV_HAIRLINE_V = 0.005
```

含义：

- `U` 从左到右铺满一条细长区域，对应 17 列锚点。
- `V` 有两行：一行给中间行，一行给发际线行。
- OBJ 里写的是 raw V，SDK 加载 OBJ 时会做 `1 - v` 翻转，所以这里的低 V 对应贴图图片的上方区域。

如果你的纹理图集里这块区域已经被占用，需要调整这些常量，然后重新运行：

```bash
python python/build_extended_obj.py
```

## v2-headext：把网格再往外扩 20 个点 (522 顶点)

v1 (`face_ext.obj`, 502 点) 只在额头方向加了一条带。v2 (`face_ext_v2.obj`, 522 点) 在 v1 基础上再加一圈侧脸外圈点，让贴图能延伸到 **太阳穴 → 颧弓 → 耳前** 一带，方便做美妆、贴花、侧脸特效。

设计与实施细节见 [PLAN_headext.md](PLAN_headext.md)。

### v2 拓扑速览

```
v2: 522 顶点 = 468 MP + 17 v1 middle + 17 v1 hairline + 10 lateral_mid + 10 lateral_out
v2: 948 三角面 = 916 v1 + 32 lateral ribbon
```

| 段 | 下标 | 数量 | 来源 |
|----|------|------|------|
| MediaPipe | `0..467` | 468 | `FaceLandmarker` |
| v1 middle | `468..484` | 17 | `build_middle_row()` |
| v1 hairline | `485..501` | 17 | `sample_hairline_lateral_extend_dense()` + `smooth_hairline_corner_aware()` |
| **lateral_mid** | `502..511` | 10 | `sample_lateral_extension()` + `lift_lateral_to_3d()` |
| **lateral_out** | `512..521` | 10 | 同上 |

lateral 顺序统一为：左 5 (top→bottom: 太阳穴→耳前) + 右 5 (top→bottom)，与 `MP_LATERAL_ANCHORS_LEFT + MP_LATERAL_ANCHORS_RIGHT` 一一对应。每侧锚点链 `[127/356, 234/454, 93/323, 132/361, 58/288]`。

### 算法管线 (v2 部分)

1. **2D 外圈检测** (`sample_lateral_extension`)：从每个 lateral MP 锚点出发，沿 face-up 的垂直方向 (左脸 -X，右脸 +X) 行进，遇到 `skin ∪ all-hair` silhouette 边界停下，停下前的最后一个像素就是 `lateral_out`。`lateral_mid` 取 `lerp(MP锚点, lateral_out, 0.5)`。
2. **椭球先验** (`fit_head_ellipsoid`)：用 468 MP 点的 XY 包围盒 + 解剖学先验 (axis_margin=1.30, depth/width=1.15, center_depth_offset=0.35) 估计一个轴对齐椭球。**不用纯算法 LSQ 拟合**，因为 468 点只覆盖头部前半层会得到 saddle 解。
3. **3D Z 解算** (`lift_lateral_to_3d`)：对每个 lateral 2D 点 `(x, y)`，把椭球方程 `((x-cx)/a)² + ((y-cy)/b)² + ((z-cz)/c)² = 1` 当作 z 的一元二次解，选靠 `z_front_sign` 一侧的根；落到 xy-envelope 外时回退最近 MP 锚点的 Z。
4. **拼装**：`assemble_full_v2` 把 468 + 17 + 17 + 10 + 10 = 522 个 (x, y, z) 顺序连起来。

### CLI

```bash
python -m python.extract_headext path/to/photo.jpg --out data/photo_v2.json
```

输出 JSON 在 v1 schema 上增量：

```json
{
  "version": "v2-headext",
  "n_total": 522,
  "layout": [
    "mp[0..468)", "middle[468..485)", "hairline[485..502)",
    "lateral_mid[502..512)", "lateral_out[512..522)"
  ],
  "points": [[x, y, z], ...],          // length 522
  "valid_hairline":        [...],       // length 17
  "valid_lateral":         [...],       // length 10
  "lateral_in_envelope":   [...],       // length 20 (mid then out)
  "ellipsoid": { "center": [...], "axes": [...], "z_front_sign": 1, "residual": 0.59 }
}
```

可调参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--landmark-backend` | `subprocess` | 同 v1 |
| `--lateral-max-walk-ratio` | `0.15` | 外圈点沿 perp 方向走出的最大像素 = 图宽 × 该值 |
| `--hairline-intermediates` | `1` | 给 `lateral_extend_dense` 用，1 → 33 点然后下采样到 17 |

### 调试三联图

```bash
python -m python.visualize_headext path/to/photo.jpg --out data/v2_overlay.png
```

输出依次：silhouette + 10 条 lateral 射线 / 522 点 + 32 条 lateral ribbon wireframe / Z 着色 (红=近 蓝=远)。

### Web 服务 (实时调参)

`/headext` 页面专门调 v2 lateral 参数 (与 `/hairline` 共享同一个 LRU 缓存)，10 个 slider：

| key | 默认 | 含义 |
|-----|------|------|
| `lateral_max_walk_ratio_x100` | 15 | 同 CLI |
| `lateral_mid_t_x100` | 50 | mid 点的内插系数 (50=居中, 80=贴近 out) |
| `use_full_silhouette` | ON | silhouette 用 `skin ∪ all-hair`；OFF 时只用面部毗邻 hair |
| `ellipsoid_axis_margin_x100` | 130 | 椭球 (a, b) 相对 MP 包围盒的放大系数 |
| `ellipsoid_depth_to_width_x100` | 115 | c = max(a, b) × 该值 / 100 |
| `ellipsoid_center_depth_offset_x100` | 35 | 椭球中心相对前脸的后移系数 |
| `ellipsoid_axes_clip_ratio_x100` | 100 | 超出 envelope 时 z 的限幅系数 |
| `show_silhouette_panel` / `show_points_panel` / `show_depth_panel` | ON | 单独勾选要渲染的面板 |

HTTP：

| 路径 | 方法 | 说明 |
|------|------|------|
| `/headext` | GET | 上传页 |
| `/headext/analyze` | POST (multipart) | 上传 + 用默认参数渲染一次 |
| `/headext/api/render` | POST (json `{stem, params}`) | 用新参数重渲染 (仅 CPU, 100~300 ms) |
| `/headext/outputs/<filename>` | GET | overlay PNG |

`/hairline` 页右上角有跳转链接；两个页面共享 stem，上传一次即可两边来回调。

### 生成 `face_ext_v2.obj`

```bash
python -m tools.gen_face_ext_v2
python python/check_ext_obj.py --all     # 校验 v1 + v2
```

`face_ext_v2.obj` 在 `face_ext.obj` 基础上追加 20 个 lateral 顶点 + 32 个三角面，前 502 个顶点 / UV / 法线 / 916 个三角面完全不变。

### v2 UV 在贴图哪个位置

为了不冲突已有贴图 (`imgs/texture0.png` 占 image y ≈ 229..460), v2 把 lateral 20 个顶点的 UV 放到当前 **未使用的图像顶部** (image y ≈ 0..115, 也就是 OBJ raw V ≈ 0.77..1.0)：

| 行 | OBJ raw V | image y (512 px) | 用途 |
|----|-----------|------------------|------|
| `UV_LATERAL_OUT_V` | 0.92 | ≈ 41 | 外圈 10 点 |
| `UV_LATERAL_MID_V` | 0.86 | ≈ 72 | 中间 10 点 |

每条横带又分左右两段 (U=0.05..0.45 左 / U=0.55..0.95 右)，每段铺 5 个锚点。常量定义在 `python/constants.py` 的 `UV_LATERAL_*`，配套函数 `lateral_uv_for(row, side, col)`。

用 `python/uv_template.py` 可以画一张当前 UV 布局图，叠到 `texture0.png` 上方便重绘贴图。

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
| `curve_offset_z(i)` | 发际线行的 Z 弧度 | 侧视时发际线太平、太凸或太陷 |
| `bulge_z(i)` | 中间行的 Z 弧度 | 额头扩展带看起来像平面或折角明显 |
| `UV_STRIP_U_MIN/MAX` | 新增区域在贴图里的横向范围 | 贴图内容横向拉伸、压缩或碰到其他图块 |
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
5. 如果 2D 点对但 3D 形状不好，调 `curve_offset_z(i)` 和 `bulge_z(i)`。
6. 如果 mesh 对但贴图错，调 UV 常量并重新生成 `face_ext.obj`。
