"""
model_loader.py
---------------
Loads real_fake_face_mobilenetv2_model.h5

Confirmed architecture (from binary inspection):
  - MobileNetV2 backbone (frozen)
  - GlobalAveragePooling2D
  - Dense(128, relu)  → dense_2
  - Dense(1, sigmoid) → dense_3   ← OUTPUT
  - Loss: binary_crossentropy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BUG FIX — Label inversion (root cause of "real shows as fake")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The model was trained on the "140k Real and Fake Faces" dataset
where the conventional class mapping may vary depending on how the checkpoint was exported.
Most current restored checkpoints in this project behave as:
  - Class 0 = REAL
  - Class 1 = FAKE

So:  sigmoid output = P(class=1) = P(FAKE) (default app behavior)
      p_real = 1.0 - sigmoid(output)

The old bug was an accidental class flip in interpretation, so real↔fake got reversed.
Now the behavior is controlled by MODEL_SIGMOID_IS_REAL env.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

GRADCAM_LAYER = "Conv_1"

# If your checkpoint outputs sigmoid = P(real), set this to True (default).
# If your checkpoint outputs sigmoid = P(fake), set this to False.
# Override via env var MODEL_SIGMOID_IS_REAL=0 or 1.
MODEL_SIGMOID_IS_REAL = os.environ.get("MODEL_SIGMOID_IS_REAL", "1") not in ("0", "false", "False", "no", "NO")


def load_keras_model(model_path: str) -> keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}.\n"
            "Place your `real_fake_face_mobilenetv2_model.h5` file in the `checkpoints/` directory, or set MODEL_PATH."
        )

    model = keras.models.load_model(model_path, compile=False)
    print(f"[Model] Loaded: {model_path}")
    print(f"[Model] Input : {model.input_shape}")
    print(f"[Model] Output: {model.output_shape}")
    print(f"[Model] Params: {model.count_params():,}")
    layer_names = [l.name for l in model.layers]
    if GRADCAM_LAYER in layer_names:
        print(f"[Model] GradCAM target '{GRADCAM_LAYER}' confirmed ✓")
    else:
        for l in reversed(model.layers):
            if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                print(f"[Model] GradCAM fallback: '{l.name}'")
                break
    return model


def get_gradcam_layer(model: keras.Model) -> str:
    layer_names = [l.name for l in model.layers]
    if GRADCAM_LAYER in layer_names:
        return GRADCAM_LAYER
    for l in reversed(model.layers):
        if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return l.name
    raise ValueError("No suitable Grad-CAM layer found.")


def predict_single(model: keras.Model, tensor: np.ndarray) -> dict:
    """
    Predict with signalled label convention.
    - MODEL_SIGMOID_IS_REAL=True: sigmoid = P(real).
    - MODEL_SIGMOID_IS_REAL=False: sigmoid = P(fake).

    p_real and p_fake are always returned consistently, and result is the higher of the two.
    """
    raw = model.predict(tensor, verbose=0)   # (1,1)
    sigmoid = float(raw[0][0])

    if MODEL_SIGMOID_IS_REAL:
        p_real = sigmoid
        p_fake = 1.0 - sigmoid
    else:
        p_fake = sigmoid
        p_real = 1.0 - sigmoid

    # class assignment uses max probability (safer than strict p_real>=0.5 when convention is unclear)
    if p_real >= p_fake:
        result = "REAL"
        confidence = p_real
    else:
        result = "FAKE"
        confidence = p_fake

    print(f"[DEBUG] MODEL_SIGMOID_IS_REAL={MODEL_SIGMOID_IS_REAL} | sigmoid={sigmoid:.4f} | p_real={p_real:.4f} | p_fake={p_fake:.4f} | result={result} | confidence={confidence:.4f}")

    return {
        "result":     result,
        "confidence": round(confidence, 4),
        "p_real":     round(p_real, 4),
        "p_fake":     round(p_fake, 4),
    }
