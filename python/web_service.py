"""Local web service for uploading a face image and visualizing hairline samples.

The default landmark backend is `subprocess`, which runs MediaPipe
FaceLandmarker in an isolated subprocess. This keeps the Web server alive
even if MediaPipe's native code crashes (a known WSL/EGL issue), while
still using MediaPipe for the 468 landmark detection that downstream 3D
mesh generation depends on.

Usage:
  python python/web_service.py
  python python/web_service.py --host 0.0.0.0 --port 18001 --device cuda
  python python/web_service.py --landmark-backend tasks      # in-process MediaPipe
  python python/web_service.py --landmark-backend solutions  # legacy mp.solutions
  python python/web_service.py --landmark-backend parsing    # no MediaPipe (fallback)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.face_landmarks import FaceLandmarker, SolutionsFaceLandmarker
    from python.face_parsing import FaceParser
    from python.hairline_2d import (
        sample_hairline_lateral_extend_dense,
        smooth_hairline_corner_aware,
    )
    from python.lift_3d import (
        assemble_full,
        build_middle_row,
        lift_hairline_to_3d,
    )
    from python._index_map_data import INDEX_MAP_468
    from python.obj_io import read_obj
else:
    from . import constants as C
    from .face_landmarks import FaceLandmarker, SolutionsFaceLandmarker
    from .face_parsing import FaceParser
    from .hairline_2d import (
        sample_hairline_lateral_extend_dense,
        smooth_hairline_corner_aware,
    )
    from .lift_3d import (
        assemble_full,
        build_middle_row,
        lift_hairline_to_3d,
    )
    from ._index_map_data import INDEX_MAP_468
    from .obj_io import read_obj


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DATA_DIR = os.path.join(PROJECT_DIR, "data", "web")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LANDMARK_BACKENDS = ("subprocess", "tasks", "solutions", "parsing")
SUBPROCESS_TIMEOUT_S = 120
CACHE_MAX_ENTRIES = 8

WSL_FRIENDLY_ENV = {
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
    "GALLIUM_DRIVER": "llvmpipe",
    "MEDIAPIPE_DISABLE_GPU": "1",
    "EGL_PLATFORM": "surfaceless",
}


# Per-strategy adjustable parameters (slider definitions). The Web UI builds
# its slider panel from this schema and re-renders via /api/render on change.
PARAM_SCHEMA: tuple[dict, ...] = (
    dict(key="intermediates", label="中间插入点数 (intermediates)",
         type="int", min=0, max=4, step=1, default=1,
         hint="点数 = 17 + 16 × N。0 = 不加密 (17 点), 1 = 33 点, 2 = 49 点, 3 = 65 点。"),
    dict(key="max_walk_ratio_x1000", label="横向延伸幅度 (max_walk_ratio × 1000)",
         type="int", min=0, max=50, step=1, default=15,
         hint="最外侧锚点沿 hit 横行向外推, 上限 = N × 0.1% 图宽。0 = 关闭横向延伸。"),
    dict(key="outer_per_side", label="横向延伸的外侧点数 (outer_per_side, -1 = 自动)",
         type="int", min=-1, max=15, step=1, default=-1,
         hint="每侧最外 N 个点应用横向走出。-1 = 自动 = (intermediates+1)×3。"),
    dict(key="density_run_length", label="hair 连续像素门限 (density_run_length)",
         type="int", min=0, max=30, step=1, default=0,
         hint="射线必须命中连续 N 个 hair 像素才算正式命中, 0 = 关闭, 越大越能滤掉单像素噪声 (但会让 hit 离锚点更远)。"),
    dict(key="smooth_iters", label="平滑迭代次数 (smooth_iters)",
         type="int", min=0, max=5, step=1, default=2,
         hint="0 = 不平滑, 越大越圆滑 (>=3 转角会被磨掉)。"),
    dict(key="corner_cos_x100", label="转角检测阈值 (cos × 100)",
         type="int", min=0, max=100, step=1, default=60,
         hint="cos(夹角) < 阈值时该点视为转角不平滑。阈值越小越严, 越大越多转角被保留。"),
    dict(key="use_adjacent", label="过滤背景 hair (use_adjacent)",
         type="bool", default=True,
         hint="开启: 只保留邻接 face skin 的 hair 像素, 过滤被错分成 hair 的背景。"),
)
PARAM_DEFAULTS: dict = {p["key"]: p["default"] for p in PARAM_SCHEMA}


def _normalize_params(raw: dict) -> dict:
    """Coerce + clamp raw JSON params to validated values."""
    out = dict(PARAM_DEFAULTS)
    for schema in PARAM_SCHEMA:
        key = schema["key"]
        if key not in raw:
            continue
        v = raw[key]
        if schema["type"] == "int":
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
            v = max(schema["min"], min(schema["max"], v))
        elif schema["type"] == "bool":
            if isinstance(v, str):
                v = v.lower() in ("1", "true", "yes", "on")
            else:
                v = bool(v)
        out[key] = v
    return out


def _params_to_kwargs(params: dict) -> dict:
    """Translate UI-shaped params into function kwargs."""
    return {
        "intermediates": params["intermediates"],
        "use_adjacent": params["use_adjacent"],
        "max_walk_ratio": params["max_walk_ratio_x1000"] / 1000.0,
        "outer_per_side": None if params["outer_per_side"] < 0 else params["outer_per_side"],
        "density_run_length": params["density_run_length"],
    }


@dataclass(frozen=True)
class AnalysisInit:
    original_name: str
    original_url: str
    stem: str
    backend: str
    prepare_ms: int
    initial_render: dict
    initial_params: dict
    param_schema: tuple


def detect_landmarks_via_subprocess(image_path: str) -> np.ndarray:
    """Run MediaPipe FaceLandmarker in a subprocess and return (468, 3) landmarks."""
    env = dict(os.environ)
    for key, value in WSL_FRIENDLY_ENV.items():
        env.setdefault(key, value)

    with tempfile.TemporaryDirectory(prefix="head3d_lmk_") as tmp:
        out_path = os.path.join(tmp, "landmarks.npy")
        cmd = [
            sys.executable,
            "-m",
            "python._mediapipe_subprocess",
            image_path,
            out_path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"MediaPipe 子进程超时 (>{SUBPROCESS_TIMEOUT_S}s), 请检查模型是否已下载或机器是否过载。"
            )

        if completed.returncode == 0 and os.path.isfile(out_path):
            return np.load(out_path)

        stderr_tail = "\n".join(
            line for line in (completed.stderr or "").strip().splitlines()[-8:]
        )
        if completed.returncode == 4:
            raise RuntimeError("MediaPipe 没检测到人脸, 请换一张更清楚的正脸图片。")
        if completed.returncode == 3:
            raise RuntimeError("MediaPipe 子进程读不到这张图片, 请确认图片格式。")
        if completed.returncode < 0:
            signal_no = -completed.returncode
            raise RuntimeError(
                "MediaPipe 子进程被信号 "
                f"{signal_no} 终止 (很可能是 WSL 下 EGL/OpenGL 段错误)。"
                "可以尝试 --landmark-backend parsing 退回到无 MediaPipe 的发际线估计。"
                + (f"\n子进程日志:\n{stderr_tail}" if stderr_tail else "")
            )
        raise RuntimeError(
            f"MediaPipe 子进程异常退出 (exit={completed.returncode})。"
            + (f"\n子进程日志:\n{stderr_tail}" if stderr_tail else "")
        )


class HairlineWebAnalyzer:
    """Caches the parser model + per-image (rgb, parse_map, landmarks).

    Re-running a single strategy with new params is then a CPU-only pass
    over the cached arrays — typically tens of milliseconds.
    """

    def __init__(self, device: str | None = None, landmark_backend: str = "subprocess"):
        if landmark_backend not in LANDMARK_BACKENDS:
            raise ValueError(f"unknown landmark backend: {landmark_backend}")
        self.device = device
        self.landmark_backend = landmark_backend
        self._parser: FaceParser | None = None
        self._lock = threading.Lock()
        self._cache: "collections.OrderedDict[str, dict]" = collections.OrderedDict()

    def _get_parser(self) -> FaceParser:
        if self._parser is None:
            self._parser = FaceParser(device=self.device)
        return self._parser

    def _detect_landmarks_inproc(self, rgb: np.ndarray) -> np.ndarray:
        if self.landmark_backend == "tasks":
            landmarker = FaceLandmarker(static_image_mode=True)
        elif self.landmark_backend == "solutions":
            landmarker = SolutionsFaceLandmarker(static_image_mode=True)
        else:
            raise RuntimeError(f"in-proc detect not supported for backend {self.landmark_backend}")
        try:
            return landmarker.detect(rgb)
        finally:
            landmarker.close()

    def _store_cache(self, stem: str, entry: dict) -> None:
        if stem in self._cache:
            self._cache.move_to_end(stem)
        self._cache[stem] = entry
        while len(self._cache) > CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)

    def prepare(self, image_path: str, stem: str) -> dict:
        """Run parse + landmark detection once and cache the result."""
        import cv2

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError("无法读取图片, 请确认文件是 jpg/png/webp 格式。")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with self._lock:
            parse_map = self._get_parser().parse(rgb)
            landmarks: np.ndarray | None = None
            if self.landmark_backend != "parsing":
                if self.landmark_backend == "subprocess":
                    landmarks = detect_landmarks_via_subprocess(image_path)
                else:
                    landmarks = self._detect_landmarks_inproc(np.ascontiguousarray(rgb))
                if landmarks is None:
                    raise RuntimeError("没有检测到人脸, 请换一张正脸或光线更清楚的图片。")
            entry = {
                "rgb": rgb,
                "parse_map": parse_map,
                "landmarks": landmarks,
            }
            self._store_cache(stem, entry)
        return entry

    def render_dense(self, stem: str, params: dict, out_dir: str) -> dict:
        """Re-run `lateral_extend_dense` with the given params and write images."""
        cache = self._cache.get(stem)
        if cache is None:
            raise KeyError("缓存里没有这张图, 请重新上传。")
        if cache["landmarks"] is None:
            raise RuntimeError("当前 backend 不返回 landmarks (parsing 模式), 无法跑 lateral_extend_dense。")

        rgb = cache["rgb"]
        parse_map = cache["parse_map"]
        landmarks = cache["landmarks"]

        started = time.perf_counter()
        with self._lock:
            self._cache.move_to_end(stem)
            kwargs = _params_to_kwargs(params)
            hairline, valid = sample_hairline_lateral_extend_dense(
                landmarks, parse_map, **kwargs
            )
            iters = params["smooth_iters"]
            if iters > 0:
                hairline = smooth_hairline_corner_aware(
                    hairline,
                    valid,
                    iterations=iters,
                    corner_cos_threshold=params["corner_cos_x100"] / 100.0,
                )

            points_filename = f"{stem}_dense_points.png"
            curve_filename = f"{stem}_dense_curve.png"
            _save_rgb(os.path.join(out_dir, points_filename), draw_hairline_points(rgb, hairline, valid))
            _save_rgb(os.path.join(out_dir, curve_filename), draw_hairline_curve(rgb, hairline, valid))
        elapsed = int((time.perf_counter() - started) * 1000)

        return {
            "points_filename": points_filename,
            "curve_filename": curve_filename,
            "n_total": int(len(valid)),
            "n_valid": int(valid.sum()),
            "elapsed_ms": elapsed,
        }

    def prepare_preview(self, stem: str, crown_lift_frac: float | None = None) -> dict:
        """Run the v1 (502-vertex) pipeline with the given crown-lift and
        return the points in both MP-order (raw output) and OBJ-vertex
        order (slot-for-slot match with face_ext.obj), plus metadata.

        crown_lift_frac:  how far above the detected hairline (in fractions
                          of face height) the mesh ribbon's top row should
                          sit. ``None`` falls back to C.HAIRLINE_CROWN_LIFT_FRAC.

        The caller is expected to have already called `prepare(image_path, stem)`.
        """
        cache = self._cache.get(stem)
        if cache is None:
            raise KeyError("缓存里没有这张图, 请重新上传。")
        if cache["landmarks"] is None:
            raise RuntimeError("当前 backend 不返回 landmarks (parsing 模式), 无法跑 preview。")

        rgb = cache["rgb"]
        parse_map = cache["parse_map"]
        landmarks = cache["landmarks"]
        h, w = rgb.shape[:2]

        with self._lock:
            self._cache.move_to_end(stem)

            hairline_dense, _ = sample_hairline_lateral_extend_dense(
                landmarks, parse_map, intermediates=1,
            )
            hairline_17 = hairline_dense[::2]
            if hairline_17.shape[0] != C.N_ANCHORS:
                hairline_17 = hairline_dense[:C.N_ANCHORS]
            valid_17 = np.ones(C.N_ANCHORS, dtype=bool)
            hairline_smoothed = smooth_hairline_corner_aware(
                hairline_17.copy(), valid_17, iterations=2,
            )
            hairline_3d = lift_hairline_to_3d(
                landmarks, hairline_smoothed,
                crown_lift_frac=crown_lift_frac,
            )
            middle_3d = build_middle_row(landmarks, hairline_3d)

            pts_mp = assemble_full(landmarks, middle_3d, hairline_3d)

            # Reorder to match the OBJ's vertex slots: for slots [0..468)
            # OBJ slot i holds the MP landmark with id INDEX_MAP_468[i].
            # Slots [468..502) are identity (face_ext.obj uses the same
            # ordering as the JSON output).
            pts_obj = np.zeros_like(pts_mp)
            for obj_idx, mp_idx in enumerate(INDEX_MAP_468):
                pts_obj[obj_idx] = pts_mp[mp_idx]
            pts_obj[C.N_MP:] = pts_mp[C.N_MP:]

        effective_lift = (
            float(crown_lift_frac)
            if crown_lift_frac is not None
            else float(C.HAIRLINE_CROWN_LIFT_FRAC)
        )
        return {
            "image": {"width": int(w), "height": int(h)},
            "n_total": int(pts_mp.shape[0]),
            "points_mp_order": pts_mp.astype(float).tolist(),
            "points_obj_order": pts_obj.astype(float).tolist(),
            "valid_hairline": valid_17.tolist(),
            "groups": {
                "mp": [0, C.N_MP],
                "v1_middle": [C.MIDDLE_START, C.HAIRLINE_START],
                "v1_hairline": [C.HAIRLINE_START, C.N_TOTAL],
            },
            "crown_lift_frac": effective_lift,
        }


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def _hairline_pixels(hairline_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = hairline_2d * np.array([width, height], dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts.astype(np.int32)


def draw_hairline_points(rgb: np.ndarray, hairline_2d: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import cv2

    out = rgb.copy()
    height, width = out.shape[:2]
    pts = _hairline_pixels(hairline_2d, width, height)
    radius = max(3, int(min(width, height) * 0.005))
    font_scale = max(0.35, min(width, height) / 1500.0)

    for idx, (x, y) in enumerate(pts):
        color = (40, 230, 90) if valid[idx] else (255, 80, 80)
        cv2.circle(out, (int(x), int(y)), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(idx + 1),
            (int(x) + radius + 2, int(y) - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, radius // 3),
            cv2.LINE_AA,
        )
    return out


def draw_hairline_curve(rgb: np.ndarray, hairline_2d: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import cv2

    out = rgb.copy()
    height, width = out.shape[:2]
    pts = _hairline_pixels(hairline_2d, width, height)
    thickness = max(3, int(min(width, height) * 0.006))

    for i in range(len(pts) - 1):
        p0 = tuple(map(int, pts[i]))
        p1 = tuple(map(int, pts[i + 1]))
        color = (40, 220, 255) if valid[i] and valid[i + 1] else (255, 90, 90)
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

    return out


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>发际线分析 (lateral_extend_dense 实时调参)</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #1f2937;
    }
    body { margin: 0; }
    .page { max-width: 1240px; margin: 0 auto; padding: 28px 20px 48px; }
    .hero, .card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 22px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .hero { padding: 24px 26px; }
    h1 { margin: 0 0 10px; font-size: 26px; }
    p { line-height: 1.6; margin: 0 0 8px; }
    .muted { color: #6b7280; font-size: 14px; }
    form { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 16px; }
    input[type=file] {
      flex: 1 1 320px;
      border: 1px dashed #9ca3af;
      border-radius: 10px;
      background: #f9fafb;
      padding: 12px;
    }
    button.primary {
      border: 0; border-radius: 10px; background: #2563eb; color: white;
      font-weight: 700; padding: 12px 20px; cursor: pointer;
    }
    button.primary:hover { background: #1d4ed8; }
    .error {
      margin-top: 16px; padding: 12px 14px; border-radius: 10px;
      background: #fef2f2; color: #991b1b; border: 1px solid #fecaca;
      white-space: pre-wrap;
    }
    .layout {
      margin-top: 22px;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
    }
    @media (max-width: 920px) {
      .layout { grid-template-columns: 1fr; }
    }
    .controls .control {
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px dashed #e5e7eb;
    }
    .controls .control:last-child { border-bottom: 0; margin-bottom: 0; padding-bottom: 0; }
    .controls label {
      display: block; font-weight: 600; font-size: 13px; margin-bottom: 6px;
    }
    .controls .row { display: flex; align-items: center; gap: 8px; }
    .controls input[type=range] { flex: 1; }
    .controls .val {
      min-width: 36px; text-align: right; font-variant-numeric: tabular-nums;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #1d4ed8; font-weight: 700;
    }
    .controls .hint { display: block; font-size: 11px; color: #6b7280; margin-top: 6px; line-height: 1.4; }
    .meta {
      margin: 8px 0 14px; padding: 10px 12px; border-radius: 10px;
      background: #eff6ff; color: #1e3a8a; border: 1px solid #bfdbfe;
      font-size: 14px;
    }
    .images {
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    }
    @media (max-width: 720px) {
      .images { grid-template-columns: 1fr; }
    }
    .images figure { margin: 0; }
    .images img {
      width: 100%; height: auto; border-radius: 12px;
      background: #111827; display: block;
    }
    .images figcaption {
      margin-top: 6px; font-size: 12px; color: #6b7280; text-align: center;
    }
    .reset {
      background: transparent; color: #6b7280; border: 1px solid #d1d5db;
      border-radius: 8px; padding: 6px 12px; font-size: 12px; cursor: pointer;
      margin-top: 6px;
    }
    .reset:hover { color: #1f2937; border-color: #6b7280; }
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 999px;
      background: #e0e7ff; color: #1e3a8a; font-size: 12px; margin-left: 4px;
      font-family: ui-monospace, monospace;
    }
    .status { font-size: 12px; color: #6b7280; margin-left: 8px; }
    .status.busy { color: #b45309; }
    .status.err  { color: #991b1b; }
    .original-strip {
      margin-top: 14px;
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    }
    .original-strip img {
      max-height: 96px; border-radius: 8px; background: #111827;
    }
    .original-strip small { color: #6b7280; }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div style="float:right;font-size:13px;">
        <a href="/preview" style="color:#2563eb;text-decoration:none;margin-left:8px;">/preview (3D 验证)</a>
      </div>
      <h1>发际线分析 · lateral_extend_dense 实时调参</h1>
      <p class="muted">上传一张正脸照片后, 在左侧滑块上调整任何参数都会立即重新渲染发际线 (parse map + landmarks 只在上传时跑一次)。</p>
      <form action="/hairline/analyze" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/png,image/jpeg,image/webp" required>
        <button class="primary" type="submit">上传 / 开始分析</button>
      </form>
      {% if error %}
      <div class="error">{{ error }}</div>
      {% endif %}
      {% if init %}
      <div class="original-strip">
        <img src="{{ init.original_url }}" alt="上传的原图">
        <small>文件 <b>{{ init.original_name }}</b> · backend <b>{{ init.backend }}</b> · 上传后 parse+landmark 用时 {{ init.prepare_ms }} ms · 缓存 stem <code>{{ init.stem }}</code></small>
      </div>
      {% endif %}
    </section>

    {% if init %}
    <section class="layout">
      <aside class="card controls">
        <h2 style="margin:0 0 12px; font-size:16px;">参数调节
          <span class="badge">lateral_extend_dense</span>
          <span class="status" id="status">--</span>
        </h2>
        <div id="ctrl_root"></div>
        <button class="reset" id="reset_btn" type="button">↺ 重置全部参数</button>
      </aside>

      <article class="card">
        <div class="meta" id="meta">点数: <b id="n_total">--</b> · valid: <b id="n_valid">--</b> · 渲染耗时: <b id="elapsed">--</b> ms</div>
        <div class="images">
          <figure>
            <img id="img_points" src="" alt="发际线点">
            <figcaption>发际线点 (绿=命中, 红=回退)</figcaption>
          </figure>
          <figure>
            <img id="img_curve" src="" alt="发际线曲线">
            <figcaption>连成的发际线曲线</figcaption>
          </figure>
        </div>
      </article>
    </section>

    <script>
    (function() {
      const STEM = {{ init.stem|tojson }};
      const SCHEMA = {{ init.param_schema|tojson }};
      const INIT_PARAMS = {{ init.initial_params|tojson }};
      const INIT_RESULT = {{ init.initial_render|tojson }};
      const DEFAULTS = Object.fromEntries(SCHEMA.map(s => [s.key, s.default]));

      const $ = id => document.getElementById(id);
      const root = $("ctrl_root");
      const status = $("status");

      function makeControl(s, value) {
        const wrap = document.createElement("div");
        wrap.className = "control";
        const label = document.createElement("label");
        label.htmlFor = "p_" + s.key;
        label.textContent = s.label;
        wrap.appendChild(label);

        const row = document.createElement("div");
        row.className = "row";
        let input;
        if (s.type === "bool") {
          input = document.createElement("input");
          input.type = "checkbox";
          input.id = "p_" + s.key;
          input.checked = !!value;
        } else {
          input = document.createElement("input");
          input.type = "range";
          input.id = "p_" + s.key;
          input.min = s.min; input.max = s.max; input.step = s.step;
          input.value = value;
        }
        row.appendChild(input);
        const valSpan = document.createElement("span");
        valSpan.className = "val";
        valSpan.id = "p_" + s.key + "_val";
        valSpan.textContent = (s.type === "bool") ? (value ? "ON" : "OFF") : String(value);
        row.appendChild(valSpan);
        wrap.appendChild(row);

        if (s.hint) {
          const hint = document.createElement("span");
          hint.className = "hint";
          hint.textContent = s.hint;
          wrap.appendChild(hint);
        }
        return { wrap, input, valSpan };
      }

      const inputs = {};
      SCHEMA.forEach(s => {
        const { wrap, input, valSpan } = makeControl(s, INIT_PARAMS[s.key]);
        root.appendChild(wrap);
        inputs[s.key] = { input, valSpan, schema: s };
      });

      function readParams() {
        const out = {};
        for (const [k, { input, schema }] of Object.entries(inputs)) {
          if (schema.type === "bool") out[k] = input.checked;
          else out[k] = parseInt(input.value, 10);
        }
        return out;
      }

      function applyResult(j) {
        const t = Date.now();
        $("img_points").src = "/hairline/outputs/" + j.points_filename + "?t=" + t;
        $("img_curve").src  = "/hairline/outputs/" + j.curve_filename  + "?t=" + t;
        $("n_total").textContent = j.n_total;
        $("n_valid").textContent = j.n_valid;
        $("elapsed").textContent = j.elapsed_ms;
      }

      let renderSeq = 0;
      let pendingTimer = null;

      async function render() {
        const seq = ++renderSeq;
        status.textContent = "渲染中…";
        status.className = "status busy";
        try {
          const r = await fetch("/hairline/api/render", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ stem: STEM, params: readParams() })
          });
          if (!r.ok) {
            const t = await r.text();
            if (seq === renderSeq) {
              status.textContent = "错误: " + t;
              status.className = "status err";
            }
            return;
          }
          const j = await r.json();
          if (seq !== renderSeq) return;
          applyResult(j);
          status.textContent = "✓ 已更新";
          status.className = "status";
        } catch (e) {
          if (seq === renderSeq) {
            status.textContent = "错误: " + e;
            status.className = "status err";
          }
        }
      }

      function scheduleRender() {
        if (pendingTimer) clearTimeout(pendingTimer);
        pendingTimer = setTimeout(() => { pendingTimer = null; render(); }, 200);
      }

      Object.entries(inputs).forEach(([k, { input, valSpan, schema }]) => {
        const refresh = () => {
          valSpan.textContent = (schema.type === "bool")
            ? (input.checked ? "ON" : "OFF")
            : input.value;
          scheduleRender();
        };
        input.addEventListener("input", refresh);
        input.addEventListener("change", refresh);
      });

      $("reset_btn").addEventListener("click", () => {
        SCHEMA.forEach(s => {
          const { input, valSpan } = inputs[s.key];
          if (s.type === "bool") {
            input.checked = !!s.default;
            valSpan.textContent = input.checked ? "ON" : "OFF";
          } else {
            input.value = s.default;
            valSpan.textContent = String(s.default);
          }
        });
        scheduleRender();
      });

      // Bootstrap with the server-rendered initial result.
      applyResult(INIT_RESULT);
      status.textContent = "✓ 初始渲染";
      status.className = "status";
    })();
    </script>
    {% endif %}
  </main>
</body>
</html>
"""


PREVIEW_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v1 端到端验证 · 502 点 + face_ext + texture0</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7fb; color: #1f2937; }
    body { margin: 0; }
    .page { max-width: 1480px; margin: 0 auto; padding: 24px 18px 40px; }
    .hero, .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05); }
    h1 { margin: 0 0 10px; font-size: 24px; }
    p { line-height: 1.6; margin: 0 0 8px; }
    .muted { color: #6b7280; font-size: 13px; }
    form { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 14px; }
    input[type=file] { flex: 1 1 320px; border: 1px dashed #9ca3af; border-radius: 10px; background: #f9fafb; padding: 12px; }
    button.primary { border: 0; border-radius: 10px; background: #2563eb; color: white; font-weight: 700; padding: 12px 20px; cursor: pointer; }
    button.primary:hover { background: #1d4ed8; }
    .error { margin-top: 14px; padding: 12px 14px; border-radius: 10px; background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; white-space: pre-wrap; }
    .layout { margin-top: 20px; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr); gap: 18px; }
    @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } }
    .panel-title { font-size: 14px; font-weight: 700; color: #374151; margin: 0 0 8px; display: flex; align-items: center; gap: 8px; }
    .panel-title small { color: #6b7280; font-weight: 500; font-size: 12px; }
    .img-wrap { position: relative; width: 100%; }
    .img-wrap img { display: block; width: 100%; height: auto; border-radius: 10px; background: #111827; }
    .img-wrap canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; font-size: 12px; color: #374151; }
    .legend .item { display: flex; align-items: center; gap: 6px; }
    .legend .swatch { width: 12px; height: 12px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.15); }
    .threejs-grid { display: grid; grid-template-rows: 1fr 1fr; gap: 14px; }
    .three-card { background: #0f172a; border-radius: 12px; overflow: hidden; position: relative; min-height: 360px; aspect-ratio: 16 / 11; }
    .three-card .three-canvas { display: block; width: 100%; height: 100%; }
    .three-card.overlay-card { background: #000; aspect-ratio: auto; min-height: 0; }
    .three-card.overlay-card .overlay-bg { display: block; width: 100%; height: auto; }
    .three-card.overlay-card .overlay-canvas {
      position: absolute; left: 0; top: 0; width: 100%; height: 100%;
      pointer-events: none;
    }
    .three-card .badge {
      position: absolute; top: 10px; left: 10px; padding: 4px 10px;
      border-radius: 999px; background: rgba(255, 255, 255, 0.92);
      color: #1f2937; font-size: 12px; font-weight: 700;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 2;
    }
    .three-card .controls-help {
      position: absolute; bottom: 10px; right: 10px; padding: 4px 9px;
      background: rgba(15, 23, 42, 0.7); color: #e5e7eb; font-size: 11px;
      border-radius: 6px; pointer-events: none;
    }
    .three-card .toggle-row {
      position: absolute; bottom: 10px; left: 10px;
      background: rgba(15, 23, 42, 0.7); color: #e5e7eb; font-size: 11px;
      border-radius: 6px; padding: 4px 8px; display: flex; gap: 8px; align-items: center;
    }
    .three-card .toggle-row label { display: flex; align-items: center; gap: 4px; cursor: pointer; }
    .three-card .toggle-row input { accent-color: #60a5fa; }
    .meta { margin: 10px 0 4px; padding: 8px 10px; border-radius: 8px; background: #eff6ff; color: #1e3a8a; border: 1px solid #bfdbfe; font-size: 12px; font-family: ui-monospace, monospace; word-break: break-all; }
    .nav-links { float: right; font-size: 13px; }
    .nav-links a { color: #2563eb; text-decoration: none; margin-left: 8px; }
    .nav-links a:hover { text-decoration: underline; }
    .status { font-size: 12px; color: #6b7280; margin-left: 8px; }
    .status.busy { color: #b45309; } .status.err { color: #991b1b; }
    .uv-tex-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
    @media (max-width: 900px) { .uv-tex-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="nav-links">
        <a href="/hairline">/hairline</a>
      </div>
      <h1>v1 hairline mesh · 端到端 3D 验证 <span class="status" id="hero_status"></span></h1>
      <p class="muted">上传一张正脸照, 左侧画 3 组识别点 (MP 468 + v1 middle 17 + v1 hairline 17 = 502), 右上是 ortho 正交投影 = 原图为底 + 活脸 mesh 贴图 overlay (与原图严格对齐), 右下是 OBJ 文件里的标准模板 mesh (可旋转). 新加的 middle / hairline 行 z 用矢状-arc 公式 z = z_anchor + dy²/(2R) 反解, 始终向头后方向偏, 严格落在头骨曲面上.</p>
      <form action="/preview/analyze" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/png,image/jpeg,image/webp" required>
        <button class="primary" type="submit">上传 / 开始分析</button>
      </form>
      {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </section>

    {% if init %}
    <section class="layout">
      <article class="card">
        <h2 class="panel-title">原图 + 502 识别点 <small>3 组色编</small></h2>
        <div class="img-wrap">
          <img id="orig_img" src="{{ init.original_url }}" alt="原图">
          <canvas id="overlay_canvas"></canvas>
        </div>
        <div class="legend">
          <div class="item"><span class="swatch" style="background:#cccccc"></span> MediaPipe 468</div>
          <div class="item"><span class="swatch" style="background:#ffa040"></span> v1 middle 17</div>
          <div class="item"><span class="swatch" style="background:#ffc864"></span> v1 hairline 17</div>
        </div>
        <div class="meta" id="meta">--</div>
        <p class="muted" style="margin-top:10px; font-size:11px; line-height:1.4;">
          crown_lift 已锁定为 6% face_h (HAIRLINE_CROWN_LIFT_FRAC), face_ext.obj 与运行时检测都用同一个常数。
        </p>
      </article>

      <article class="card">
        <h2 class="panel-title">3D 渲染对比 <small>OBJ + texture0.png</small></h2>
        <div class="threejs-grid">
          <div class="three-card overlay-card" id="card_live">
            <span class="badge">活脸 overlay · 原图 + 贴图 mesh</span>
            <img class="overlay-bg" id="overlay_bg" src="{{ init.original_url }}" alt="overlay bg">
            <canvas class="three-canvas overlay-canvas" id="canvas_live"></canvas>
            <div class="toggle-row">
              <label><input type="checkbox" id="wf_live"> 线框</label>
              <label><input type="checkbox" id="tex_live" checked> 贴图</label>
              <label>不透明度 <input type="range" id="alpha_live" min="0" max="100" value="100" style="width:80px"></label>
            </div>
            <span class="controls-help">静态 ortho · 与原图严格对齐</span>
          </div>
          <div class="three-card" id="card_canonical">
            <span class="badge">canonical · face_ext.obj (可旋转)</span>
            <canvas class="three-canvas" id="canvas_canonical"></canvas>
            <div class="toggle-row">
              <label><input type="checkbox" id="wf_canonical"> 线框</label>
              <label><input type="checkbox" id="tex_canonical" checked> 贴图</label>
            </div>
            <span class="controls-help">drag · wheel · right-drag</span>
          </div>
        </div>
      </article>
    </section>

    <section class="card" style="margin-top:18px;">
      <h2 class="panel-title">UV 拓扑底图 & 自定义贴图 <small>美术拿这张图去画, 上传 PNG 即可预览效果</small></h2>
      <div class="uv-tex-grid">
        <div>
          <div style="background:#fff; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden;">
            <img id="uv_template_img" src="/preview/assets/uv_template.png" alt="UV 拓扑底图" style="display:block; width:100%; height:auto;">
          </div>
          <div style="margin-top:10px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; font-size:12px;">
            <a id="uv_download" class="primary" style="text-decoration:none; padding:7px 14px; background:#2563eb; color:#fff; border-radius:8px; font-weight:600;" href="/preview/assets/uv_template.png" download="uv_template.png">下载 PNG</a>
            <label style="display:flex; gap:6px; align-items:center; cursor:pointer;">
              <input type="checkbox" id="uv_overlay_toggle"> 叠加当前 texture0 对比
            </label>
            <span class="muted">512x512 · face_ext.obj 的 UV 拓扑参考</span>
          </div>
          <p class="muted" style="margin-top:8px; font-size:11px; line-height:1.5;">
            分区色编与下方贴图一致: MP 468 浅灰, v1 middle 17 橙, v1 hairline 17 黄.
            V_raw=1 对应图片顶部 (Three.js flipY=true). 美术保持画布 512x512 即可.
          </p>
        </div>
        <div>
          <h3 style="margin:0 0 8px; font-size:13px; color:#374151;">
            上传自定义贴图 <span id="tex_status" class="status">默认 texture0.png</span>
          </h3>
          <input type="file" id="tex_upload" accept="image/png,image/jpeg,image/webp"
                 style="width:100%; box-sizing:border-box; border:1px dashed #9ca3af; border-radius:10px; background:#f9fafb; padding:12px;">
          <div style="margin-top:8px; display:flex; gap:10px; flex-wrap:wrap;">
            <button id="tex_reset" type="button"
                    style="border:1px solid #cbd5e1; background:#fff; color:#1f2937; border-radius:8px; padding:7px 14px; cursor:pointer; font-weight:600;">
              恢复默认 texture0
            </button>
          </div>
          <div style="margin-top:12px;">
            <div class="muted" style="font-size:12px;">当前贴图预览</div>
            <img id="tex_current_preview" src="/preview/assets/texture0.png" alt="current texture"
                 style="display:block; max-width:100%; height:auto; margin-top:6px; border:1px solid #e5e7eb; border-radius:8px; background:#111827;">
          </div>
          <p class="muted" style="margin-top:8px; font-size:11px; line-height:1.5;">
            上传后右上 ortho overlay 与右下 canonical viewer 同步切换. 仅当前浏览器会话生效, 不会覆盖 imgs/texture0.png.
          </p>
        </div>
      </div>
    </section>

    <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }
    }
    </script>
    <script type="module">
      import * as THREE from 'three';
      import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

      const STEM = {{ init.stem|tojson }};
      const heroStatus = document.getElementById('hero_status');

      function setStatus(text, cls) {
        heroStatus.textContent = text;
        heroStatus.className = 'status' + (cls ? ' ' + cls : '');
      }

      // --- Fetch detected points -----------------------------------------
      // crown_lift is now locked to C.HAIRLINE_CROWN_LIFT_FRAC server-side
      // (the slider was removed). The page fetches the detected mesh once
      // per upload and never re-issues with a different lift.
      // textureMeshRefs collects each viewer's { material, getTex, setTex }
      // so the upload handler can hot-swap textures on both viewers at once.
      let currentData = null;
      const textureMeshRefs = [];

      async function fetchData() {
        const r = await fetch('/preview/api/data/' + STEM);
        if (!r.ok) throw new Error('GET /preview/api/data failed: ' + r.status);
        return await r.json();
      }

      let dataPromise = (async () => {
        setStatus('载入识别点 …', 'busy');
        currentData = await fetchData();
        return currentData;
      })();

      // --- Left: original image + 502 colored dots overlay ---------------
      const origImg = document.getElementById('orig_img');
      const overlayCanvas = document.getElementById('overlay_canvas');
      const metaEl = document.getElementById('meta');

      function redrawDotsAndMeta(data) {
        if (!data) return;
        if (!(origImg.complete && origImg.naturalWidth > 0)) {
          origImg.addEventListener('load', () => redrawDotsAndMeta(data), { once: true });
          return;
        }
        const W = origImg.naturalWidth, H = origImg.naturalHeight;
        overlayCanvas.width = W; overlayCanvas.height = H;
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, W, H);
        const colors = { mp: '#cccccc', v1_middle: '#ffa040', v1_hairline: '#ffc864' };
        const radii  = { mp: 1.6,        v1_middle: 3.2,        v1_hairline: 3.2 };
        const pts = data.points_mp_order;
        for (const [name, [a, b]] of Object.entries(data.groups)) {
          const c = colors[name];
          if (!c) continue;
          ctx.fillStyle = c;
          const r = (radii[name] || 2) * (Math.min(W, H) / 600.0);
          for (let i = a; i < b; i++) {
            const p = pts[i];
            ctx.beginPath();
            ctx.arc(p[0] * W, p[1] * H, r, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
        const ANCHORS = [127,234,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454];
        const HAIR_START = 485;
        let anchor_z_sum = 0, hair_z_sum = 0;
        for (let i = 0; i < 17; i++) {
          anchor_z_sum += pts[ANCHORS[i]][2];
          hair_z_sum   += pts[HAIR_START + i][2];
        }
        const dz_mean = (hair_z_sum - anchor_z_sum) / 17;
        const lift_pct = (data.crown_lift_frac * 100).toFixed(1);
        metaEl.innerHTML =
          '图片 ' + W + 'x' + H + ' · 502 点 (468 MP + 17 mid + 17 hair)<br>' +
          'valid_hairline ' + data.valid_hairline.filter(Boolean).length + '/17 · ' +
          'crown_lift = ' + lift_pct + '% face_h · ' +
          '⟨ z(hairline) − z(MP anchor) ⟩ = ' + dz_mean.toFixed(4) +
          (dz_mean > 0 ? ' ✓' : ' ✗');
      }

      dataPromise.then(redrawDotsAndMeta).catch(e => setStatus('左侧渲染失败: ' + e, 'err'));

      // --- UV template image: clean ⇄ overlay (with current texture0) ----
      const uvImg      = document.getElementById('uv_template_img');
      const uvDownload = document.getElementById('uv_download');
      document.getElementById('uv_overlay_toggle').addEventListener('change', e => {
        const path = e.target.checked
          ? '/preview/assets/uv_template_overlay.png'
          : '/preview/assets/uv_template.png';
        const filename = e.target.checked ? 'uv_template_overlay.png' : 'uv_template.png';
        uvImg.src = path;
        uvDownload.href = path;
        uvDownload.setAttribute('download', filename);
      });

      // --- Texture hot-swap pipeline -------------------------------------
      // Each viewer (top ortho overlay + bottom canonical) registers its
      // material into textureMeshRefs after init. swapTextureURL(url) loads
      // a new THREE.Texture, points every registered material at it, and
      // disposes the previous one. The same pipeline serves both the
      // upload-button path and the reset-to-default button.
      async function loadTextureURL(url) {
        const t = await new Promise((res, rej) => {
          new THREE.TextureLoader().load(url, res, undefined, rej);
        });
        t.colorSpace = THREE.SRGBColorSpace;
        t.flipY = true;
        t.needsUpdate = true;
        return t;
      }

      const texStatus  = document.getElementById('tex_status');
      const texPreview = document.getElementById('tex_current_preview');

      async function swapTextureURL(url, label) {
        texStatus.textContent = '载入贴图…'; texStatus.className = 'status busy';
        try {
          const newTex = await loadTextureURL(url);
          for (const ref of textureMeshRefs) {
            const old = ref.material.map;
            ref.material.map = newTex;
            ref.material.needsUpdate = true;
            if (old && old !== newTex) old.dispose();
          }
          texPreview.src = url;
          texStatus.textContent = label || '✓ 已切换';
          texStatus.className = 'status';
        } catch (e) {
          texStatus.textContent = '错误: ' + e.message;
          texStatus.className = 'status err';
        }
      }

      document.getElementById('tex_upload').addEventListener('change', async e => {
        const file = e.target.files[0];
        if (!file) return;
        texStatus.textContent = '上传中…'; texStatus.className = 'status busy';
        try {
          const fd = new FormData();
          fd.append('texture', file);
          const r = await fetch('/preview/api/texture', { method: 'POST', body: fd });
          if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + (await r.text()));
          const j = await r.json();
          await swapTextureURL(j.url, '✓ 已切换到 ' + file.name);
        } catch (err) {
          texStatus.textContent = '上传失败: ' + err.message;
          texStatus.className = 'status err';
        }
      });

      document.getElementById('tex_reset').addEventListener('click', () => {
        document.getElementById('tex_upload').value = '';
        swapTextureURL('/preview/assets/texture0.png', '已恢复 texture0.png');
      });

      // --- Right: TOP = ortho overlay on photo, BOTTOM = 3D canonical ----
      // Geometry source: we ship an indexed JSON instead of using OBJLoader
      // because OBJLoader expands every face into 3 unique vertices,
      // breaking the OBJ-vertex ↔ detected-point mapping. The JSON
      // deduplicates by (pos_idx, uv_idx, normal_idx) triple and ships a
      // parallel `buffer_to_obj_v` array so we can override positions
      // per-OBJ-vertex.
      //
      // UV convention: uv_template.py maps img_y = (1 - V_raw) × size, so
      // V_raw=1 corresponds to the TOP of the texture image. Three.js
      // with default texture.flipY=true samples V=1 = top of image, so no
      // UV flip is needed.
      //
      // Axis convention: face.obj uses Y growing DOWN and +Z = back of
      // head (nose tip z≈-0.15, ear-pre z≈+0.28).
      //
      //   - Bottom (canonical) viewer uses a perspective camera with
      //     Y up + camera-looks-down -Z; mesh.scale=(1,-1,-1) flips Y
      //     and Z and preserves winding (two negatives = det +1).
      //
      //   - Top (live) viewer uses an orthographic frustum with top=0,
      //     bottom=1 (i.e. y-down image coords). Camera at +Z looks down
      //     -Z so face.obj +Z (back of head) lands behind front of face
      //     for depth ordering. No mesh scale is applied; we keep the
      //     image-normalized x/y exactly so the overlay registers with
      //     the original photo pixel-for-pixel. Material uses DoubleSide
      //     because the y-inverted ortho flips triangle winding.
      const geomJsonURL = '/preview/assets/face_ext.json';
      const texURL = '/preview/assets/texture0.png';

      function buildBaseMaterial(texture) {
        return new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
          color: 0xffffff,
        });
      }

      async function loadIndexedGeometry() {
        const j = await fetch(geomJsonURL).then(r => {
          if (!r.ok) throw new Error('GET ' + geomJsonURL + ' → ' + r.status);
          return r.json();
        });
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(j.positions, 3));
        geom.setAttribute('uv', new THREE.Float32BufferAttribute(j.uvs, 2));
        geom.setAttribute('normal', new THREE.Float32BufferAttribute(j.normals, 3));
        geom.setIndex(j.indices);
        return { geom, bufferToObj: j.buffer_to_obj_v, nObjVertices: j.n_obj_vertices };
      }

      function applyDetectedPositions(geometry, bufferToObj, points_obj_order) {
        // points_obj_order: length n_obj_vertices, each [x, y, z] in face.obj
        // local space (same axes as face.obj canonical positions).
        const pos = geometry.attributes.position;
        if (pos.count !== bufferToObj.length) {
          throw new Error('bufferToObj/position length mismatch: ' +
            pos.count + ' vs ' + bufferToObj.length);
        }
        for (let i = 0; i < pos.count; i++) {
          const objIdx = bufferToObj[i];
          const p = points_obj_order[objIdx];
          if (!p) continue;
          pos.setXYZ(i, p[0], p[1], p[2]);
        }
        pos.needsUpdate = true;
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
      }

      function applyFlatPositions(geometry, bufferToObj, flatPositions) {
        // flatPositions: length n_obj_vertices * 3, OBJ vertex order.
        const pos = geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) {
          const o = bufferToObj[i] * 3;
          pos.setXYZ(i, flatPositions[o], flatPositions[o + 1], flatPositions[o + 2]);
        }
        pos.needsUpdate = true;
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
      }

      function setupScene(canvasEl) {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);

        // Soft ambient so the unlit MeshBasicMaterial isn't needed but
        // helpful if we toggle to MeshStandard later.
        scene.add(new THREE.AmbientLight(0xffffff, 0.9));
        const dir = new THREE.DirectionalLight(0xffffff, 0.4);
        dir.position.set(1, 1, 2);
        scene.add(dir);

        const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
        camera.position.set(0, 0, 1.6);

        const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: false });
        renderer.setPixelRatio(window.devicePixelRatio || 1);

        const controls = new OrbitControls(camera, canvasEl);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.rotateSpeed = 0.9;
        controls.zoomSpeed = 0.9;
        controls.target.set(0.5, 0.55, 0.0);

        function resize() {
          const r = canvasEl.getBoundingClientRect();
          renderer.setSize(r.width, r.height, false);
          camera.aspect = Math.max(1e-3, r.width / Math.max(1e-3, r.height));
          camera.updateProjectionMatrix();
        }
        resize();
        const ro = new ResizeObserver(resize);
        ro.observe(canvasEl);

        function loop() {
          controls.update();
          renderer.render(scene, camera);
          requestAnimationFrame(loop);
        }
        requestAnimationFrame(loop);
        return { scene, camera, controls, renderer };
      }

      function centerAndFit(mesh, camera, controls) {
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const radius = Math.max(size.x, size.y, size.z) * 0.6;
        controls.target.copy(center);
        const dir = new THREE.Vector3(0, 0, 1).normalize();
        camera.position.copy(center).addScaledVector(dir, radius / Math.tan((camera.fov * Math.PI / 180) / 2) * 1.15);
        camera.near = radius * 0.01;
        camera.far  = radius * 100.0;
        camera.updateProjectionMatrix();
        controls.update();
      }

      async function setupCanonicalViewer(canvasId) {
        const canvasEl = document.getElementById(canvasId);
        const { scene, camera, controls, renderer } = setupScene(canvasEl);

        const { geom, bufferToObj, nObjVertices } = await loadIndexedGeometry();
        const tex = await new Promise((resolve, reject) => {
          new THREE.TextureLoader().load(texURL, resolve, undefined, reject);
        });
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.flipY = true;
        tex.needsUpdate = true;

        const matTex = buildBaseMaterial(tex);
        const matWire = new THREE.MeshBasicMaterial({
          color: 0x55ff88, wireframe: true, transparent: true, opacity: 0.6,
        });
        textureMeshRefs.push({ material: matTex });

        const mesh = new THREE.Mesh(geom, matTex);
        mesh.scale.set(1, -1, -1);

        const wire = new THREE.Mesh(geom, matWire);
        wire.scale.copy(mesh.scale);
        wire.visible = false;
        scene.add(mesh);
        scene.add(wire);

        document.getElementById('wf_canonical').addEventListener('change', e => { wire.visible = e.target.checked; });
        document.getElementById('tex_canonical').addEventListener('change', e => {
          mesh.visible = e.target.checked;
          if (!e.target.checked) wire.visible = true;
        });

        centerAndFit(mesh, camera, controls);
      }

      async function setupLiveOverlay(canvasId, bgImgId) {
        const canvasEl = document.getElementById(canvasId);
        const bgImg = document.getElementById(bgImgId);
        const data = await dataPromise;

        const scene = new THREE.Scene();  // no background → canvas stays transparent

        // y-down image coords: top=0, bottom=1, so world y=0 lands at the
        // TOP of the screen. Camera at +Z looking -Z so face.obj +Z (back
        // of head) ends up FURTHER from the camera = behind the front.
        const camera = new THREE.OrthographicCamera(0, 1, 0, 1, -10, 10);
        camera.position.set(0, 0, 1);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({
          canvas: canvasEl, antialias: true, alpha: true, premultipliedAlpha: true,
        });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setClearColor(0x000000, 0);

        const { geom, bufferToObj, nObjVertices } = await loadIndexedGeometry();
        const tex = await new Promise((resolve, reject) => {
          new THREE.TextureLoader().load(texURL, resolve, undefined, reject);
        });
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.flipY = true;
        tex.needsUpdate = true;

        const have = data.points_obj_order ? data.points_obj_order.length : 0;
        if (have !== nObjVertices) {
          console.warn('detected points length ' + have +
            ' != OBJ vertex count ' + nObjVertices + ', rendering overlap only.');
        }
        applyDetectedPositions(geom, bufferToObj, data.points_obj_order);

        const matTex = new THREE.MeshBasicMaterial({
          map: tex, side: THREE.DoubleSide, color: 0xffffff,
          transparent: true, opacity: 1.0,
        });
        const matWire = new THREE.MeshBasicMaterial({
          color: 0x55ff88, wireframe: true, transparent: true, opacity: 0.7,
          depthTest: false,
        });
        textureMeshRefs.push({ material: matTex });

        const mesh = new THREE.Mesh(geom, matTex);
        const wire = new THREE.Mesh(geom, matWire);
        wire.visible = false;
        scene.add(mesh);
        scene.add(wire);

        document.getElementById('wf_live').addEventListener('change', e => { wire.visible = e.target.checked; });
        document.getElementById('tex_live').addEventListener('change', e => { mesh.visible = e.target.checked; });
        document.getElementById('alpha_live').addEventListener('input', e => {
          matTex.opacity = parseInt(e.target.value, 10) / 100;
        });

        function resize() {
          const r = bgImg.getBoundingClientRect();
          if (r.width < 1 || r.height < 1) return;
          renderer.setSize(r.width, r.height, false);
        }
        if (bgImg.complete && bgImg.naturalWidth > 0) resize();
        else bgImg.addEventListener('load', resize, { once: true });
        const ro = new ResizeObserver(resize);
        ro.observe(bgImg);

        function loop() {
          renderer.render(scene, camera);
          requestAnimationFrame(loop);
        }
        requestAnimationFrame(loop);
      }

      Promise.all([
        setupLiveOverlay('canvas_live', 'overlay_bg'),
        setupCanonicalViewer('canvas_canonical'),
      ]).then(() => setStatus('✓ 已加载', null))
        .catch(e => { console.error(e); setStatus('3D 加载失败: ' + e.message, 'err'); });
    </script>
    {% endif %}
  </main>
</body>
</html>
"""


def create_app(device: str | None = None, landmark_backend: str = "subprocess"):
    from flask import (
        Flask,
        jsonify,
        redirect,
        render_template_string,
        request,
        send_from_directory,
    )
    from werkzeug.utils import secure_filename

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    analyzer = HairlineWebAnalyzer(device=device, landmark_backend=landmark_backend)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)

    # All hairline-detection endpoints live under /hairline/* so future
    # features (texture / 3D mesh / hair-style transfer / ...) can claim their
    # own top-level namespaces without colliding.
    @app.get("/")
    def root_index():
        return redirect("/hairline", code=302)

    @app.get("/hairline")
    def hairline_index():
        return render_template_string(INDEX_HTML, init=None, error=None)

    @app.post("/hairline/analyze")
    def hairline_analyze():
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            return render_template_string(INDEX_HTML, init=None, error="请选择一张图片。"), 400
        if not allowed_file(upload.filename):
            return render_template_string(
                INDEX_HTML, init=None, error="只支持 jpg、jpeg、png、webp 图片。",
            ), 400

        safe_name = secure_filename(upload.filename)
        ext = os.path.splitext(safe_name)[1].lower()
        stem = f"{uuid.uuid4().hex}_{os.path.splitext(safe_name)[0]}"
        original_filename = stem + ext
        original_path = os.path.join(WEB_DATA_DIR, original_filename)
        upload.save(original_path)

        started = time.perf_counter()
        try:
            analyzer.prepare(original_path, stem)
            initial_render = analyzer.render_dense(stem, dict(PARAM_DEFAULTS), WEB_DATA_DIR)
        except Exception as exc:
            return render_template_string(INDEX_HTML, init=None, error=str(exc)), 500
        prepare_ms = int((time.perf_counter() - started) * 1000)

        init = AnalysisInit(
            original_name=safe_name,
            original_url=f"/hairline/outputs/{original_filename}",
            stem=stem,
            backend=analyzer.landmark_backend,
            prepare_ms=prepare_ms,
            initial_render=initial_render,
            initial_params=dict(PARAM_DEFAULTS),
            param_schema=PARAM_SCHEMA,
        )
        return render_template_string(INDEX_HTML, init=init, error=None)

    @app.post("/hairline/api/render")
    def hairline_api_render():
        payload = request.get_json(silent=True) or {}
        stem = payload.get("stem")
        if not isinstance(stem, str) or not stem:
            return ("missing stem", 400)
        raw_params = payload.get("params") or {}
        if not isinstance(raw_params, dict):
            return ("params must be an object", 400)
        params = _normalize_params(raw_params)
        try:
            result = analyzer.render_dense(stem, params, WEB_DATA_DIR)
        except KeyError as exc:
            return (str(exc), 410)
        except Exception as exc:
            return (str(exc), 500)
        return jsonify(result)

    @app.get("/hairline/outputs/<path:filename>")
    def hairline_outputs(filename: str):
        return send_from_directory(WEB_DATA_DIR, filename)

    # ----- /preview: end-to-end 3D verification -----------------------------
    # Caches per (stem, crown_lift) the most recent /preview/api/data payload
    # so slider drags don't keep re-running the pipeline at known values.
    preview_data_cache: "collections.OrderedDict[tuple[str, float | None], dict]" = collections.OrderedDict()

    def _store_preview_data(key: "tuple[str, float | None]", data: dict) -> None:
        if key in preview_data_cache:
            preview_data_cache.move_to_end(key)
        preview_data_cache[key] = data
        while len(preview_data_cache) > CACHE_MAX_ENTRIES:
            preview_data_cache.popitem(last=False)

    @app.get("/preview")
    def preview_index():
        return render_template_string(PREVIEW_HTML, init=None, error=None)

    @app.post("/preview/analyze")
    def preview_analyze():
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            return render_template_string(PREVIEW_HTML, init=None, error="请选择一张图片。"), 400
        if not allowed_file(upload.filename):
            return render_template_string(
                PREVIEW_HTML, init=None, error="只支持 jpg、jpeg、png、webp 图片。",
            ), 400

        safe_name = secure_filename(upload.filename)
        ext = os.path.splitext(safe_name)[1].lower()
        stem = f"{uuid.uuid4().hex}_{os.path.splitext(safe_name)[0]}"
        original_filename = stem + ext
        original_path = os.path.join(WEB_DATA_DIR, original_filename)
        upload.save(original_path)

        started = time.perf_counter()
        try:
            analyzer.prepare(original_path, stem)
            data = analyzer.prepare_preview(stem)
        except Exception as exc:
            return render_template_string(PREVIEW_HTML, init=None, error=str(exc)), 500
        prepare_ms = int((time.perf_counter() - started) * 1000)
        _store_preview_data((stem, None), data)

        init = AnalysisInit(
            original_name=safe_name,
            original_url=f"/preview/outputs/{original_filename}",
            stem=stem,
            backend=analyzer.landmark_backend,
            prepare_ms=prepare_ms,
            initial_render={},
            initial_params={},
            param_schema=(),
        )
        return render_template_string(PREVIEW_HTML, init=init, error=None)

    def _parse_crown_lift(arg: str | None) -> float | None:
        """Parse ?crown_lift_x100=NN query arg, value × 1/100 → fraction."""
        if arg is None:
            return None
        try:
            v = int(arg)
        except (TypeError, ValueError):
            return None
        v = max(0, min(40, v))
        return v / 100.0

    @app.get("/preview/api/data/<stem>")
    def preview_api_data(stem: str):
        lift = _parse_crown_lift(request.args.get("crown_lift_x100"))
        cache_key = (stem, lift)
        cached = preview_data_cache.get(cache_key)
        if cached is not None:
            preview_data_cache.move_to_end(cache_key)
            return jsonify(cached)
        try:
            data = analyzer.prepare_preview(stem, crown_lift_frac=lift)
        except KeyError as exc:
            return (str(exc), 410)
        except Exception as exc:
            return (str(exc), 500)
        _store_preview_data(cache_key, data)
        return jsonify(data)

    @app.get("/preview/outputs/<path:filename>")
    def preview_outputs(filename: str):
        return send_from_directory(WEB_DATA_DIR, filename)

    @app.get("/preview/assets/face_ext.obj")
    def preview_asset_obj():
        return send_from_directory(PROJECT_DIR, "face_ext.obj", mimetype="text/plain")

    @app.get("/preview/assets/face_ext.json")
    def preview_asset_obj_json():
        """Indexed-geometry JSON, ready to drop into Three.js BufferGeometry.

        OBJLoader expands faces into 3 × num_faces non-indexed vertices,
        which loses the OBJ-vertex-to-buffer-vertex mapping that we need
        to override positions per OBJ vertex. This endpoint deduplicates
        by the (pos_idx, uv_idx, normal_idx) triple so the client can
        build an indexed BufferGeometry and keep a parallel
        ``buffer_to_obj_v`` mapping for live-position updates.
        """
        cache = getattr(preview_asset_obj_json, "_cache", None)
        if cache is None:
            mesh = read_obj(os.path.join(PROJECT_DIR, "face_ext.obj"))
            triple_to_buf: dict[tuple[int, int, int], int] = {}
            positions: list[float] = []
            uvs: list[float] = []
            normals: list[float] = []
            buffer_to_obj_v: list[int] = []
            indices: list[int] = []
            for face in mesh.faces:
                for vi, ti, ni in face:
                    key = (vi, ti, ni)
                    buf_idx = triple_to_buf.get(key)
                    if buf_idx is None:
                        buf_idx = len(buffer_to_obj_v)
                        triple_to_buf[key] = buf_idx
                        x, y, z = mesh.positions[vi]
                        positions.extend((x, y, z))
                        if ti >= 0 and ti < len(mesh.texcoords):
                            u, v = mesh.texcoords[ti]
                        else:
                            u, v = 0.0, 0.0
                        uvs.extend((u, v))
                        if ni >= 0 and ni < len(mesh.normals):
                            nx, ny, nz = mesh.normals[ni]
                        else:
                            nx, ny, nz = 0.0, 0.0, 1.0
                        normals.extend((nx, ny, nz))
                        buffer_to_obj_v.append(vi)
                    indices.append(buf_idx)

            cache = {
                "positions": positions,
                "uvs": uvs,
                "normals": normals,
                "indices": indices,
                "buffer_to_obj_v": buffer_to_obj_v,
                "n_buffer_vertices": len(buffer_to_obj_v),
                "n_obj_vertices": len(mesh.positions),
                "n_faces": len(mesh.faces),
            }
            preview_asset_obj_json._cache = cache  # type: ignore[attr-defined]
        return jsonify(cache)

    @app.get("/preview/assets/canonical_positions")
    def preview_asset_canonical_positions():
        """Return the 502 canonical positions for face_ext.obj computed at
        a given crown_lift fraction. Lets the right-bottom canonical viewer
        follow the same slider as the runtime detected mesh, instead of being
        frozen at the build-time HAIRLINE_CROWN_LIFT_FRAC.

        Query: ?crown_lift_x100=NN (0..40). Missing → C.HAIRLINE_CROWN_LIFT_FRAC.
        Returns: { "positions": [502*3 floats, OBJ vertex order],
                   "crown_lift_frac": float }
        """
        lift = _parse_crown_lift(request.args.get("crown_lift_x100"))
        if lift is None:
            lift = float(C.HAIRLINE_CROWN_LIFT_FRAC)

        cache_key = ("canonical_positions", lift)
        cached = preview_data_cache.get(cache_key)
        if cached is not None:
            preview_data_cache.move_to_end(cache_key)
            return jsonify(cached)

        # Re-run the same arc + face-up-lift formula as build_extended_obj,
        # in-memory, without rewriting the OBJ. canonical_extension_positions
        # is the single source of truth shared with the build script.
        if __package__ is None or __package__ == "":
            from python.build_extended_obj import (
                build_inverse_index_map,
                canonical_extension_positions,
            )
        else:
            from .build_extended_obj import (
                build_inverse_index_map,
                canonical_extension_positions,
            )

        base_mesh = read_obj(os.path.join(PROJECT_DIR, "face.obj"))
        inv = build_inverse_index_map()
        middle, hairline = canonical_extension_positions(base_mesh, inv, crown_lift_frac=lift)

        positions: list[float] = []
        for x, y, z in base_mesh.positions:           # 468 MP vertices
            positions.extend((float(x), float(y), float(z)))
        for row in (middle, hairline):                # 17 + 17 extension vertices
            for x, y, z in row:
                positions.extend((float(x), float(y), float(z)))

        result = {"positions": positions, "crown_lift_frac": float(lift)}
        _store_preview_data(cache_key, result)
        return jsonify(result)

    @app.get("/preview/assets/texture0.png")
    def preview_asset_texture():
        return send_from_directory(os.path.join(PROJECT_DIR, "imgs"), "texture0.png")

    @app.get("/preview/assets/uv_template.png")
    def preview_asset_uv_template():
        return send_from_directory(os.path.join(PROJECT_DIR, "imgs"), "uv_template.png")

    @app.get("/preview/assets/uv_template_overlay.png")
    def preview_asset_uv_template_overlay():
        return send_from_directory(os.path.join(PROJECT_DIR, "imgs"), "uv_template_overlay.png")

    _UPLOAD_TEX_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    _UPLOAD_TEX_MAX_BYTES = 16 * 1024 * 1024  # 16 MB

    @app.post("/preview/api/texture")
    def preview_api_texture_upload():
        """Receive a user-painted effect texture and stash it in WEB_DATA_DIR.

        Returns ``{ url, filename }``. The page swaps both viewers' material
        maps to this URL — session-only, never overwrites imgs/texture0.png.
        Files share WEB_DATA_DIR with photo uploads and are reachable via
        the existing ``/preview/outputs/<filename>`` route.
        """
        f = request.files.get("texture")
        if f is None or not f.filename:
            return ("missing 'texture' file part", 400)
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in _UPLOAD_TEX_EXTS:
            return (f"unsupported format {ext!r}; use PNG/JPG/WebP", 400)
        # Content-Length is a hint; werkzeug also enforces MAX_CONTENT_LENGTH
        # if set on the app — we keep a soft cap by streaming to disk and
        # truncating if it exceeds the limit, then rejecting.
        name = f"tex_{uuid.uuid4().hex}{ext}"
        path = os.path.join(WEB_DATA_DIR, name)
        f.save(path)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > _UPLOAD_TEX_MAX_BYTES:
            try:
                os.remove(path)
            except OSError:
                pass
            return (f"file too large: {size} bytes > {_UPLOAD_TEX_MAX_BYTES}", 413)
        return jsonify({
            "url": f"/preview/outputs/{name}",
            "filename": name,
            "size": size,
        })

    @app.get("/health")
    def health():
        return {"ok": True, "backend": analyzer.landmark_backend, "cached": list(analyzer._cache.keys())}

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        return render_template_string(
            INDEX_HTML, init=None, error="图片太大了, 当前限制是 20 MB。",
        ), 413

    @app.get("/favicon.ico")
    def favicon():
        return redirect("data:,")

    app.config["PARAM_SCHEMA"] = PARAM_SCHEMA
    app.config["PARAM_DEFAULTS"] = PARAM_DEFAULTS

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local hairline analysis web service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--device", default=os.environ.get("HEAD3D_DEVICE"))
    parser.add_argument(
        "--landmark-backend",
        choices=list(LANDMARK_BACKENDS),
        default=os.environ.get("HEAD3D_LANDMARK_BACKEND", "subprocess"),
        help=(
            "subprocess (default): MediaPipe in an isolated subprocess, survives native crashes; "
            "tasks: in-process MediaPipe Tasks; "
            "solutions: in-process legacy mp.solutions.face_mesh; "
            "parsing: no MediaPipe, hairline estimated from face parsing only."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--threaded",
        action="store_true",
        help="Enable Flask threaded request handling. Off by default for stability.",
    )
    args = parser.parse_args()

    app = create_app(device=args.device, landmark_backend=args.landmark_backend)
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
        threaded=args.threaded,
    )


if __name__ == "__main__":
    main()
