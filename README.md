# Forensic AI v3 — MobileNetV2 Image Authenticity Analyser

## What changed from v2

| Issue | Fix |
|---|---|
| Real images predicted as FAKE | Root cause found: model outputs `P(fake)`, not `P(real)`. Fixed in `model_loader.py` |
| LIME reasoning had raw numbers | Rewritten with natural-language spatial analysis — no weight numbers in output |
| LIME reasoning was same every time | `random_state=None` + pooled descriptor phrases — reasoning differs per image |
| All on one page | 3-page flow: Upload → Analyse (loading) → Results |
| Dark theme | Full light theme with clean card layout |
| Only JPEG/PNG | Supports JPEG, PNG, BMP, WebP, TIFF, GIF (any Pillow format) |

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

## Flow

```
/ (upload.html)  →  /analyse (analyse.html)  →  /results (results.html)
     │                     │                           │
  Pick file          POST /api/predict           Show verdict
  Store b64          Animate steps              Grad-CAM heatmap
  → navigate         Store JSON result          LIME overlay + masks
                     → navigate                 Reasoning report
```

## Model Convention (confirmed)

```
real_fake_face_mobilenetv2_model.h5
  └─ Dense(1, sigmoid)  ← output
  └─ Loss: binary_crossentropy
  └─ Convention: label 1 = REAL  →  sigmoid output = P(real)
  └─ P(fake) = 1 - P(real)
```
- Default app behavior assumes `sigmoid = P(real)`.
- If your model outputs `sigmoid = P(fake)` (inverted), run as:
  - `MODEL_SIGMOID_IS_REAL=0 python app.py` (Linux/macOS)
  - PowerShell: `$env:MODEL_SIGMOID_IS_REAL=0; python app.py`
  - cmd: `set MODEL_SIGMOID_IS_REAL=0 && python app.py`
- If your model uses the opposite convention (label 1 = FAKE / sigmoid=P(fake)), set env var: `MODEL_SIGMOID_IS_REAL=0` before `python app.py`.

## Supported Image Formats

JPEG · PNG · BMP · WebP · TIFF · GIF (first frame) · any Pillow-readable format

## API

```
POST /api/predict
  field: image (multipart)
  returns: {
    result, confidence, p_real, p_fake,
    original_b64, gradcam_b64,
    lime_overlay, lime_positive, lime_negative,
    lime_reasoning
  }

GET /api/health
```
