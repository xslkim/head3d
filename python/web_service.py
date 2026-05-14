"""Local web service for uploading a face image and visualizing hairline samples.

Usage:
  python python/web_service.py
  python python/web_service.py --host 0.0.0.0 --port 8000 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.face_landmarks import FaceLandmarker
    from python.face_parsing import FaceParser
    from python.hairline_2d import sample_hairline, smooth_hairline
else:
    from . import constants as C
    from .face_landmarks import FaceLandmarker
    from .face_parsing import FaceParser
    from .hairline_2d import sample_hairline, smooth_hairline


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DATA_DIR = os.path.join(PROJECT_DIR, "data", "web")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class AnalysisResult:
    original_name: str
    original_url: str
    points_url: str
    curve_url: str
    valid_count: int
    total_count: int
    elapsed_ms: int


class HairlineWebAnalyzer:
    """Caches the heavy models and serializes inference calls."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._parser: FaceParser | None = None
        self._landmarker: FaceLandmarker | None = None
        self._lock = threading.Lock()

    def _get_parser(self) -> FaceParser:
        if self._parser is None:
            self._parser = FaceParser(device=self.device)
        return self._parser

    def _get_landmarker(self) -> FaceLandmarker:
        if self._landmarker is None:
            self._landmarker = FaceLandmarker(static_image_mode=True)
        return self._landmarker

    def analyze(self, image_path: str, points_path: str, curve_path: str) -> tuple[int, int]:
        import cv2

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError("无法读取图片，请确认文件是 jpg/png/webp 格式。")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            landmarks = self._get_landmarker().detect(rgb)
            if landmarks is None:
                raise RuntimeError("没有检测到人脸，请换一张正脸或光线更清楚的图片。")

            parse_map = self._get_parser().parse(rgb)
            hairline_2d, valid = sample_hairline(landmarks, parse_map)
            hairline_2d = smooth_hairline(hairline_2d, valid)

        _save_rgb(points_path, draw_hairline_points(rgb, hairline_2d, valid))
        _save_rgb(curve_path, draw_hairline_curve(rgb, hairline_2d, valid))
        return int(valid.sum()), int(valid.shape[0])


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
    radius = max(4, int(min(width, height) * 0.006))
    font_scale = max(0.4, min(width, height) / 1400.0)

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

    for i in range(C.N_ANCHORS - 1):
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
  <title>发际线分析</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #1f2937;
    }
    body { margin: 0; }
    .page { max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }
    .hero {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 12px 36px rgba(15, 23, 42, 0.08);
    }
    h1 { margin: 0 0 10px; font-size: 30px; }
    p { line-height: 1.7; }
    .muted { color: #6b7280; }
    form { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-top: 22px; }
    input[type=file] {
      flex: 1 1 320px;
      border: 1px dashed #9ca3af;
      border-radius: 12px;
      background: #f9fafb;
      padding: 14px;
    }
    button {
      border: 0;
      border-radius: 12px;
      background: #2563eb;
      color: white;
      font-weight: 700;
      padding: 14px 22px;
      cursor: pointer;
    }
    button:hover { background: #1d4ed8; }
    .error {
      margin-top: 18px;
      padding: 14px 16px;
      border-radius: 12px;
      background: #fef2f2;
      color: #991b1b;
      border: 1px solid #fecaca;
    }
    .result-meta {
      margin-top: 18px;
      padding: 12px 14px;
      border-radius: 12px;
      background: #eff6ff;
      color: #1e3a8a;
      border: 1px solid #bfdbfe;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin-top: 24px;
    }
    .card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .card h2 { margin: 0 0 10px; font-size: 18px; }
    .card img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
      background: #111827;
    }
    .legend { margin-top: 10px; font-size: 14px; color: #6b7280; }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>发际线分析</h1>
      <p class="muted">上传一张正脸照片，点击“开始分析”后，会使用当前项目的 MediaPipe + face parsing 管线识别 17 个发际线采样点，并生成两张可视化结果。</p>
      <form action="/analyze" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/png,image/jpeg,image/webp" required>
        <button type="submit">开始分析</button>
      </form>
      {% if error %}
      <div class="error">{{ error }}</div>
      {% endif %}
      {% if result %}
      <div class="result-meta">
        文件：{{ result.original_name }}；
        命中发际线点：{{ result.valid_count }}/{{ result.total_count }}；
        耗时：{{ result.elapsed_ms }} ms
      </div>
      {% endif %}
    </section>

    {% if result %}
    <section class="grid">
      <article class="card">
        <h2>原图</h2>
        <img src="{{ result.original_url }}" alt="上传的原图">
      </article>
      <article class="card">
        <h2>发际线点</h2>
        <img src="{{ result.points_url }}" alt="画出发际线采样点的图片">
        <div class="legend">绿色点表示射线命中 hair/hat；红色点表示没有命中，使用几何外推。</div>
      </article>
      <article class="card">
        <h2>发际线</h2>
        <img src="{{ result.curve_url }}" alt="画出发际线曲线的图片">
        <div class="legend">青色线段连接有效点；红色线段包含至少一个回退点。</div>
      </article>
    </section>
    {% endif %}
  </main>
</body>
</html>
"""


def create_app(device: str | None = None):
    from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
    from werkzeug.utils import secure_filename

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    analyzer = HairlineWebAnalyzer(device=device)
    os.makedirs(WEB_DATA_DIR, exist_ok=True)

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML, result=None, error=None)

    @app.post("/analyze")
    def analyze():
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            return render_template_string(INDEX_HTML, result=None, error="请选择一张图片。"), 400
        if not allowed_file(upload.filename):
            return render_template_string(
                INDEX_HTML,
                result=None,
                error="只支持 jpg、jpeg、png、webp 图片。",
            ), 400

        safe_name = secure_filename(upload.filename)
        ext = os.path.splitext(safe_name)[1].lower()
        stem = f"{uuid.uuid4().hex}_{os.path.splitext(safe_name)[0]}"
        original_filename = stem + ext
        points_filename = stem + "_points.png"
        curve_filename = stem + "_curve.png"

        original_path = os.path.join(WEB_DATA_DIR, original_filename)
        points_path = os.path.join(WEB_DATA_DIR, points_filename)
        curve_path = os.path.join(WEB_DATA_DIR, curve_filename)
        upload.save(original_path)

        started = time.perf_counter()
        try:
            valid_count, total_count = analyzer.analyze(original_path, points_path, curve_path)
        except Exception as exc:
            return render_template_string(INDEX_HTML, result=None, error=str(exc)), 500

        result = AnalysisResult(
            original_name=safe_name,
            original_url=url_for("outputs", filename=original_filename),
            points_url=url_for("outputs", filename=points_filename),
            curve_url=url_for("outputs", filename=curve_filename),
            valid_count=valid_count,
            total_count=total_count,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return render_template_string(INDEX_HTML, result=result, error=None)

    @app.get("/outputs/<path:filename>")
    def outputs(filename: str):
        return send_from_directory(WEB_DATA_DIR, filename)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        return render_template_string(
            INDEX_HTML,
            result=None,
            error="图片太大了，当前限制是 20 MB。",
        ), 413

    @app.get("/favicon.ico")
    def favicon():
        return redirect("data:,")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local hairline analysis web service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=os.environ.get("HEAD3D_DEVICE"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(device=args.device)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
