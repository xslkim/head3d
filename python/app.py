"""Flask web service: upload a face photo, extend the mesh, preview live.

Pipeline per docs/新算法.md:
  upload -> detect 468 landmarks -> deform 486-vertex mesh -> 2D overlay
  sliders (t1/t2) -> recompute extension -> re-render overlay + 3D obj

The 2D overlay is rendered server-side (CPU). The 3D view is the same
deformed OBJ streamed to a three.js viewer in the browser.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from python.deform import deform_positions
from python.landmarks import FaceLandmarker, to_pixels
from python.obj_io import ObjMesh, dumps_obj, read_obj
from python.render2d import render_overlay

app = Flask(__name__, static_folder=None)

# --- static assets loaded once ---
TEMPLATE = read_obj(os.path.join(ROOT, "face_ext.obj"))
TPL_TEXCOORDS = np.array(TEMPLATE.texcoords)[:, :2]
TPL_FACES = TEMPLATE.faces
TEX_PATH = os.path.join(ROOT, "imgs", "texture0.png")
TEXTURE = cv2.cvtColor(cv2.imread(TEX_PATH, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)

# Lazily-created detector (init can be slow / noisy).
_detector: FaceLandmarker | None = None

# In-memory session store: id -> {landmarks_px, photo_rgb, w, h}
SESSIONS: dict[str, dict] = {}


def get_detector() -> FaceLandmarker:
    global _detector
    if _detector is None:
        _detector = FaceLandmarker()
    return _detector


def _png_data_uri(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/png;base64," + b64


def _build(session: dict) -> dict:
    """Deform + render; return overlay + obj payload (parameters are fixed)."""
    positions = deform_positions(session["landmarks_px"])

    overlay = render_overlay(
        session["photo_rgb"], positions, TPL_TEXCOORDS, TPL_FACES, TEXTURE, alpha=1.0
    )

    mesh = ObjMesh()
    mesh.positions = [tuple(p) for p in positions]
    mesh.texcoords = list(TEMPLATE.texcoords)
    mesh.normals = list(TEMPLATE.normals)
    mesh.faces = TPL_FACES
    obj_text = dumps_obj(mesh, header=["# deformed head mesh"])

    return {"overlay": _png_data_uri(overlay), "obj": obj_text}


@app.route("/")
def index():
    return send_from_directory(os.path.join(ROOT, "web"), "index.html")


@app.route("/texture")
def texture():
    return send_file(TEX_PATH, mimetype="image/png")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no image"}), 400
    data = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "bad image"}), 400
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    norm = get_detector().detect(rgb)
    if norm is None:
        return jsonify({"error": "未检测到人脸 / no face detected"}), 422

    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "landmarks_px": to_pixels(norm, w, h),
        "photo_rgb": rgb,
        "w": w,
        "h": h,
    }
    payload = _build(SESSIONS[sid])
    payload.update(
        {
            "session": sid,
            "width": w,
            "height": h,
            "photo": _png_data_uri(rgb),
        }
    )
    return jsonify(payload)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18001)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
