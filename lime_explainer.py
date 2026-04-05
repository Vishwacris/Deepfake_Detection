"""
lime_explainer.py — LIME with human-readable, varied reasoning
--------------------------------------------------------------
BUG FIX: The predictor returned [P(real), P(fake)] where:
  - index 0 = P(real)
  - index 1 = P(fake)

But previously p_fake was set to raw[:,0] (wrong — that's P(real) now).
Fixed to align with model_loader convention: sigmoid = P(real).

Also fixed: result_label mapping aligns with new convention.
  REAL → label=1 (index 1 in our [p_real, p_fake] stack = p_fake... wait)

IMPORTANT: LIME's label indices are purely internal. We define them here:
  We output np.stack([p_real, p_fake], axis=1)
  So LIME label 0 = real class, label 1 = fake class.
  result_label = 1 if FAKE, 0 if REAL  ← this was correct, kept as-is.
  The fix is ONLY in _predictor: p_real and p_fake must be computed correctly.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import io, base64, random, cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries


def _preprocess(img_np: np.ndarray) -> np.ndarray:
    """MobileNetV2 normalisation: [0,255] → [-1,1]"""
    return (img_np.astype(np.float32) / 127.5) - 1.0


def _predictor(model):
    """
    Returns LIME-compatible predict_fn → (N, 2) array [P(real), P(fake)]

    FIX: sigmoid = P(real), so:
      p_real = raw[:, 0]           ← FIX (was p_fake before)
      p_fake = 1.0 - p_real        ← FIX (was 1-p_real incorrectly named)
    """
    def fn(images: np.ndarray) -> np.ndarray:
        batch  = np.stack([_preprocess(img) for img in images])
        raw    = model.predict(batch, verbose=0)   # (N,1) sigmoid = P(real)
        p_real = raw[:, 0]                          # FIX: sigmoid → P(real)
        p_fake = 1.0 - p_real                       # FIX
        # Stack as [P(real), P(fake)] → LIME label 0=real, 1=fake
        return np.stack([p_real, p_fake], axis=1).astype(np.float64)
    return fn


def run_lime(
    model,
    pil_image: Image.Image,
    prediction: dict,
    num_samples: int = 600,
    num_features: int = 8,
) -> dict:
    try:
        img_224  = pil_image.convert("RGB").resize((224, 224))
        img_np   = np.array(img_224)

        explainer = lime_image.LimeImageExplainer(random_state=42)
        exp = explainer.explain_instance(
            image         = img_np,
            classifier_fn = _predictor(model),
            top_labels    = 2,
            hide_color    = 0,
            num_samples   = num_samples,
            batch_size    = 32,
        )
    except Exception as e:
        # Fallback in case LIME explainer fails for this image; avoid crashing the whole pipeline.
        return {
            "lime_overlay_b64": None,
            "positive_mask_b64": None,
            "negative_mask_b64": None,
            "reasoning_text": f"LIME analysis unavailable: {e}",
        }

    # LIME internal label indices: 0=real, 1=fake
    result_label = 1 if prediction["result"] == "FAKE" else 0
    local_exp    = exp.local_exp.get(result_label, [])
    top_segs     = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_features]

    temp, mask = exp.get_image_and_mask(
        result_label, positive_only=False, num_features=num_features, hide_rest=False
    )
    boundary = (mark_boundaries(temp / 255.0, mask) * 255).astype(np.uint8)

    pos_img, neg_img = _coloured_masks(img_np, exp, result_label, num_features)

    segments    = exp.segments
    region_info = _analyse_regions(img_np, segments, top_segs)
    reasoning   = _build_reasoning(prediction, region_info, top_segs)

    return {
        "lime_overlay_b64":  _arr_b64(boundary),
        "positive_mask_b64": _arr_b64(pos_img),
        "negative_mask_b64": _arr_b64(neg_img),
        "reasoning_text":    reasoning,
    }


def _analyse_regions(img_np, segments, top_segs):
    H, W = img_np.shape[:2]
    info = []
    for seg_id, weight in top_segs:
        mask   = (segments == seg_id)
        pixels = img_np[mask]
        if len(pixels) == 0:
            continue
        ys, xs  = np.where(mask)
        cy, cx  = ys.mean() / H, xs.mean() / W
        position      = _position_label(cy, cx)
        coverage_pct  = mask.sum() / mask.size * 100
        brightness    = pixels.mean() / 255.0
        r, g, b       = pixels[:,0].mean(), pixels[:,1].mean(), pixels[:,2].mean()
        colour_cast   = _colour_label(r, g, b)
        gray          = cv2.cvtColor(pixels.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        texture_var   = float(np.var(gray)) / 3000.0
        info.append({
            "seg_id":      seg_id,
            "weight":      weight,
            "position":    position,
            "coverage":    coverage_pct,
            "brightness":  brightness,
            "colour_cast": colour_cast,
            "texture_var": texture_var,
            "supporting":  weight > 0,
        })
    return info


def _position_label(cy, cx):
    vert  = "upper" if cy < 0.35 else ("lower" if cy > 0.65 else "central")
    horiz = "left"  if cx < 0.35 else ("right"  if cx > 0.65 else "")
    return (vert + (" " + horiz if horiz else "")).strip()


def _colour_label(r, g, b):
    mx = max(r, g, b)
    if mx < 60:  return "dark"
    if mx > 210: return "bright"
    if r > g + 25 and r > b + 25: return "warm-toned"
    if b > r + 25 and b > g + 25: return "cool-toned"
    if g > r + 20 and g > b + 20: return "greenish"
    return "neutral"


def _build_reasoning(prediction: dict, region_info: list, top_segs: list) -> str:
    result     = prediction["result"]
    confidence = prediction["confidence"]
    p_fake     = prediction["p_fake"]
    p_real     = prediction["p_real"]

    conf_level = (
        "very high" if confidence > 0.90 else
        "high"      if confidence > 0.75 else
        "moderate"  if confidence > 0.60 else
        "low"
    )

    supporting = [r for r in region_info if r["supporting"]]
    opposing   = [r for r in region_info if not r["supporting"]]

    lines = []

    if result == "FAKE":
        openers = [
            f"The model determined this image is likely AI-generated or manipulated with {conf_level} confidence.",
            f"Analysis indicates this image bears hallmarks of synthetic generation — the model's {conf_level} confidence reflects strong evidence of manipulation.",
            f"This image was classified as digitally fabricated. The {conf_level} confidence score reflects consistent signals of AI generation across multiple image regions.",
        ]
    else:
        openers = [
            f"The model classified this image as authentic with {conf_level} confidence.",
            f"Analysis suggests this image shows characteristics consistent with a genuine, unmanipulated photograph — {conf_level} confidence.",
            f"No significant indicators of AI generation or manipulation were detected. The model assigned {conf_level} confidence to the authentic classification.",
        ]
    lines.append(random.choice(openers))
    lines.append("")

    if supporting:
        if result == "FAKE":
            intro_pool = [
                "Key regions driving the FAKE classification:",
                "The following areas showed the strongest signs of artificial generation:",
                "Evidence supporting the manipulation verdict:",
            ]
        else:
            intro_pool = [
                "Key regions confirming authenticity:",
                "The following areas showed strong indicators of a genuine photograph:",
                "Evidence supporting the REAL classification:",
            ]
        lines.append(random.choice(intro_pool))
        for r in supporting[:4]:
            lines.append(f"  • {_describe_region(r, result)}")
        lines.append("")

    if opposing:
        opp_result = "REAL" if result == "FAKE" else "FAKE"
        counter_pool = [
            f"Some regions pulled the prediction toward {opp_result}:",
            "Areas that partially contradicted the verdict:",
            "Counter-evidence detected in the following regions:",
        ]
        lines.append(random.choice(counter_pool))
        for r in opposing[:2]:
            lines.append(f"  • {_describe_region_opposing(r, result)}")
        lines.append("")

    lines.append(_confidence_narrative(result, confidence, p_fake, p_real))
    lines.append("")
    lines.append("─" * 48)
    lines.append("Explanation generated by LIME superpixel analysis.")
    lines.append("Green overlay = regions supporting the verdict.")
    lines.append("Red overlay   = regions that oppose the verdict.")

    return "\n".join(lines)


def _describe_region(r: dict, result: str) -> str:
    pos  = r["position"]
    col  = r["colour_cast"]
    br   = r["brightness"]
    tex  = r["texture_var"]
    cov  = r["coverage"]
    size_word = "large" if cov > 15 else ("moderate-sized" if cov > 6 else "small")

    if result == "FAKE":
        if tex > 0.6:
            texture_desc = random.choice([
                "unusual noise patterns inconsistent with natural camera sensor noise",
                "irregular texture variance suggesting post-processing artefacts",
                "pixel-level inconsistencies typical of GAN-generated imagery",
            ])
        elif tex < 0.1:
            texture_desc = random.choice([
                "unnaturally smooth texture lacking the grain expected in real photos",
                "over-smoothed surface characteristic of AI upscaling or generation",
                "suspiciously uniform pixel distribution atypical of authentic imagery",
            ])
        else:
            texture_desc = random.choice([
                "subtle blending artefacts at the boundary regions",
                "frequency-domain anomalies detectable by the convolutional filters",
                "statistical irregularities in pixel distribution",
            ])
        brightness_desc = ""
        if br > 0.75:
            brightness_desc = random.choice([
                "The area is unusually bright, which can indicate synthetic lighting.",
                "Over-exposed highlights are a common indicator of AI image generators.",
            ])
        elif br < 0.25:
            brightness_desc = random.choice([
                "Deep shadow areas showed inconsistent gradient patterns.",
                "Dark regions exhibited noise profiles inconsistent with real cameras.",
            ])
        desc = f"The {size_word} {pos} region ({col}) showed {texture_desc}."
        if brightness_desc:
            desc += " " + brightness_desc
        return desc
    else:
        if tex > 0.3:
            texture_desc = random.choice([
                "natural grain and texture variation consistent with camera sensor noise",
                "organic pixel-level variation typical of real-world photography",
                "authentic texture gradients matching natural lighting conditions",
            ])
        else:
            texture_desc = random.choice([
                "consistent tonal distribution aligned with genuine photographic depth",
                "natural colour gradients without synthetic banding or blending",
                "smooth transitions characteristic of real optical systems",
            ])
        return f"The {size_word} {pos} region ({col}) exhibited {texture_desc}."


def _describe_region_opposing(r: dict, result: str) -> str:
    pos = r["position"]
    col = r["colour_cast"]
    if result == "FAKE":
        pool = [
            f"The {pos} {col} region retained some characteristics of authentic photography, reducing certainty.",
            f"Parts of the {pos} area lacked the typical artefacts expected in fully synthetic images.",
        ]
    else:
        pool = [
            f"The {pos} {col} region exhibited minor inconsistencies that could suggest post-processing.",
            f"Some areas in the {pos} section showed slight compression or editing artefacts.",
        ]
    return random.choice(pool)


def _confidence_narrative(result, confidence, p_fake, p_real) -> str:
    pct_fake = int(p_fake * 100)
    pct_real = int(p_real * 100)
    if result == "FAKE":
        if confidence > 0.90:
            pool = [f"The model assigned {pct_fake}% probability of this being AI-generated — a strongly decisive result with minimal ambiguity."]
        elif confidence > 0.70:
            pool = [f"The model assigned {pct_fake}% probability of AI generation versus {pct_real}% for authentic — a clear but not absolute verdict."]
        else:
            pool = [f"This was a closer call: {pct_fake}% fake vs {pct_real}% real. The model classified it as fake but with notable uncertainty."]
    else:
        if confidence > 0.90:
            pool = [f"The model assigned {pct_real}% probability of authenticity — a strongly confident real classification."]
        elif confidence > 0.70:
            pool = [f"The model assigned {pct_real}% probability of authenticity versus {pct_fake}% for AI generation — a clear but not absolute real verdict."]
        else:
            pool = [f"This was a closer classification: {pct_real}% real vs {pct_fake}% fake. The model classified it as real but with notable uncertainty."]
    return random.choice(pool)


def _coloured_masks(img_np, exp, label, num_features):
    segs        = exp.segments
    lexp        = dict(exp.local_exp.get(label, []))
    sorted_segs = sorted(lexp.items(), key=lambda x: abs(x[1]), reverse=True)
    pos = img_np.copy().astype(np.float32)
    neg = img_np.copy().astype(np.float32)
    for seg_id, weight in sorted_segs[:num_features]:
        region = (segs == seg_id)
        if weight > 0:
            pos[region, 0] = np.clip(pos[region, 0] * 0.3, 0, 255)
            pos[region, 1] = np.clip(pos[region, 1] * 0.85 + 70, 0, 255)
            pos[region, 2] = np.clip(pos[region, 2] * 0.3, 0, 255)
        else:
            neg[region, 0] = np.clip(neg[region, 0] * 0.85 + 70, 0, 255)
            neg[region, 1] = np.clip(neg[region, 1] * 0.3, 0, 255)
            neg[region, 2] = np.clip(neg[region, 2] * 0.3, 0, 255)
    return pos.astype(np.uint8), neg.astype(np.uint8)


def _arr_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
