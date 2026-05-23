"""Local web service for uploading a face image and visualizing hairline samples.

The default landmark backend is `subprocess`, which runs MediaPipe
FaceLandmarker in an isolated subprocess. This keeps the Web server alive
even if MediaPipe's native code crashes (a known WSL/EGL issue), while
still using MediaPipe for the 468 landmark detection that downstream 3D
mesh generation depends on.

Usage:
  python python/web_service.py
  python python/web_service.py --host 0.0.0.0 --port 8000 --device cuda
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
    from python import constants as C  # noqa: F401  (kept for future use)
    from python.face_landmarks import FaceLandmarker, SolutionsFaceLandmarker
    from python.face_parsing import FaceParser
    from python.hairline_2d import (
        sample_hairline_lateral_extend_dense,
        smooth_hairline_corner_aware,
    )
else:
    from . import constants as C  # noqa: F401
    from .face_landmarks import FaceLandmarker, SolutionsFaceLandmarker
    from .face_parsing import FaceParser
    from .hairline_2d import (
        sample_hairline_lateral_extend_dense,
        smooth_hairline_corner_aware,
    )


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

    # Make the schema/params helpers available to tests via app context.
    app.config["PARAM_SCHEMA"] = PARAM_SCHEMA
    app.config["PARAM_DEFAULTS"] = PARAM_DEFAULTS

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local hairline analysis web service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
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
