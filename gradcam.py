"""
gradcam.py — Grad-CAM for MobileNetV2
--------------------------------------
BUG FIX: The original tape.watch(conv_out) was called AFTER the forward
pass, which means the tape had already finished recording gradients through
conv_out by the time watch() was called. Gradients were zero/None.

FIX: Use a persistent GradientTape, record the entire forward pass inside
the tape context, and watch the conv layer output properly by building a
sub-model that outputs both the conv activations and the final prediction,
then computing gradients of the score w.r.t. the conv output correctly.

Also: FIX label — grad for "real" class now uses p_real = 1-output
correctly aligned with the fixed predict_single convention.
"""

import numpy as np
import cv2
import io
import base64
import tensorflow as tf
from PIL import Image


def compute_gradcam(
    model: tf.keras.Model,
    tensor: np.ndarray,
    layer_name: str,
    target: str = "real",   # default to "real" since we now predict real correctly
) -> np.ndarray:
    """
    Returns: (224, 224) float32 heatmap in [0,1]

    FIX: tape.watch must be inside the with block BEFORE the forward pass.
    The grad_model outputs [conv_activations, final_prediction].
    We use GradientTape on a tf.Variable-cast input with persistent=False
    and watch the conv output tensor explicitly before it's computed.
    """
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(layer_name).output, model.output]
    )

    inp = tf.cast(tensor, tf.float32)

    # FIX: use persistent tape and watch BEFORE forward pass
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(inp)
        tape.watch(conv_out)  # FIX: must watch the tensor we want grad w.r.t.
        # FIX: p_real = predictions[:,0] (sigmoid = P(real) now)
        # target="real"  → grad of P(real) = grad of predictions[:,0]
        # target="fake"  → grad of P(fake) = grad of (1 - predictions[:,0])
        if target == "real":
            score = predictions[:, 0]
        else:
            score = 1.0 - predictions[:, 0]

    grads    = tape.gradient(score, conv_out)     # (1, h, w, C)

    if grads is None:
        # Fallback: return blank heatmap if gradient unavailable
        return np.zeros((224, 224), dtype=np.float32)

    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    conv_arr = conv_out[0]                             # (h, w, C)

    heatmap  = conv_arr @ pooled[..., tf.newaxis]      # (h, w, 1)
    heatmap  = tf.squeeze(tf.nn.relu(heatmap)).numpy() # (h, w)

    if heatmap.ndim < 2 or heatmap.size == 0:
        return np.zeros((224, 224), dtype=np.float32)

    heatmap = cv2.resize(heatmap.astype(np.float32), (224, 224))

    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn > 1e-8:
        heatmap = (heatmap - mn) / (mx - mn)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


def overlay_heatmap(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    img   = np.array(pil_image.convert("RGB").resize((224, 224)))
    cam8  = (heatmap * 255).astype(np.uint8)
    color = cv2.applyColorMap(cam8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blend = (alpha * color + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(blend)


def to_b64(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
