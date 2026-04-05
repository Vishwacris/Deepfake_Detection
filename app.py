"""
app.py — Forensic AI Flask Backend
------------------------------------
Endpoints:
  GET  /                → upload page
  GET  /analyse         → analysis (loading) page
  GET  /results         → results page
  POST /api/predict     → JSON prediction API
  GET  /api/health      → health check

Supports: JPEG, PNG, BMP, WebP, TIFF, GIF (first frame), and
          any Pillow-readable format.
"""

import os, io, traceback, base64
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from model_loader   import load_keras_model, get_gradcam_layer, predict_single
from gradcam        import compute_gradcam, overlay_heatmap, to_b64
from lime_explainer import run_lime

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "checkpoints", "real_fake_face_mobilenetv2_model.h5")
)
PORT = int(os.environ.get("PORT", 5000))

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="frontend", static_folder="static")
CORS(app)

# ── Load model ────────────────────────────────────────────────────────────────
print("=" * 56)
print("  Forensic AI  |  MobileNetV2  |  P(fake) corrected")
print("=" * 56)
MODEL = None
CONV_LAYER = None
try:
    MODEL      = load_keras_model(MODEL_PATH)
    CONV_LAYER = get_gradcam_layer(MODEL)
    print(f"  Grad-CAM layer : {CONV_LAYER}")
    print("=" * 56)
except FileNotFoundError as fe:
    print("\nERROR:\n", fe)
    print("\nHint: Ensure the model file exists at the path shown above, e.g. `checkpoints/real_fake_face_mobilenetv2_model.h5`.")
    print("Continuing with model unavailable; /api/predict will return 503 until model is loaded.")
except Exception as e:
    print("\nERROR: Failed to initialize model:", e)
    print("Continuing with model unavailable; /api/predict will return 503 until model is loaded.")


# ── Image loading — supports every Pillow-readable format ─────────────────────
ALLOWED_MIME = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp",
    "image/webp", "image/tiff", "image/gif", "image/x-ms-bmp",
    "image/x-bmp",
}

def load_any_image(file_bytes: bytes) -> Image.Image:
    """
    Opens any Pillow-readable image, handles:
      - EXIF rotation
      - Palette / RGBA → RGB conversion
      - GIF: extracts first frame
      - TIFF multi-page: first page
    """
    pil = Image.open(io.BytesIO(file_bytes))

    # GIF / animated → first frame
    if getattr(pil, "is_animated", False):
        pil.seek(0)

    # EXIF orientation correction
    pil = ImageOps.exif_transpose(pil)

    # Ensure RGB
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    return pil


def preprocess(pil: Image.Image) -> np.ndarray:
    """Resize to 224×224 and apply MobileNetV2 normalisation [-1, 1]."""
    img = pil.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, 0)   # (1, 224, 224, 3)


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def upload_page():
    return send_from_directory("frontend", "upload.html")

@app.route("/analyse")
def analyse_page():
    return send_from_directory("frontend", "analyse.html")

@app.route("/results")
def results_page():
    return send_from_directory("frontend", "results.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    if MODEL is None:
        return jsonify({
            "status": "model_missing",
            "message": "Model not loaded. Place model at checkpoints/real_fake_face_mobilenetv2_model.h5 or set MODEL_PATH.",
            "model_path": Path(MODEL_PATH).as_posix(),
        }), 503

    return jsonify({"status": "ok", "conv_layer": CONV_LAYER, "model": Path(MODEL_PATH).name})


@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Cannot predict."}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    # Accept any image mimetype Pillow can handle
    mime = f.content_type or ""
    ext  = Path(f.filename).suffix.lower()
    if mime not in ALLOWED_MIME and ext not in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"
    }:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    try:
        raw_bytes = f.read()
        pil       = load_any_image(raw_bytes)

        # Encode original (resized to 512 max dim) for display
        display_pil = pil.copy()
        display_pil.thumbnail((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        display_pil.save(buf, format="JPEG", quality=88)
        original_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        tensor = preprocess(pil)

        # ── Prediction ────────────────────────────────────────────────────────
        pred = predict_single(MODEL, tensor)

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        # FIX: target aligned with corrected label convention (sigmoid = P(real))
        # "real" → gradient of P(real); "fake" → gradient of P(fake) = 1-P(real)
        gc_target = "real" if pred["result"] == "REAL" else "fake"
        heatmap   = compute_gradcam(MODEL, tensor, CONV_LAYER, target=gc_target)
        gc_pil    = overlay_heatmap(pil, heatmap, alpha=0.48)
        gc_b64    = to_b64(gc_pil)

        # ── LIME ─────────────────────────────────────────────────────────────
        try:
            lime_out = run_lime(
                model       = MODEL,
                pil_image   = pil,
                prediction  = pred,
                num_samples = 500,
                num_features= 8,
            )
        except Exception as e:
            traceback.print_exc()
            lime_out = {
                "lime_overlay_b64": None,
                "positive_mask_b64": None,
                "negative_mask_b64": None,
                "reasoning_text": f"LIME explanation failed: {e}",
            }

        return jsonify({
            "result":          pred["result"],
            "confidence":      pred["confidence"],
            "p_real":          pred["p_real"],
            "p_fake":          pred["p_fake"],
            "original_b64":    original_b64,
            "gradcam_b64":     gc_b64,
            "lime_overlay":    lime_out["lime_overlay_b64"],
            "lime_positive":   lime_out["positive_mask_b64"],
            "lime_negative":   lime_out["negative_mask_b64"],
            "lime_reasoning":  lime_out["reasoning_text"],
            "filename":        f.filename,
            "error":           None,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n🚀  http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=False)
