import base64
import io

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from PIL import Image

from config import Config
from model import load_model as _load_trained_model
from preprocessing import preprocess_from_bytes


def load_model():
    """Charge le modèle en pilotant tout depuis Config (checkpoint, num_classes, device)
    -- évite que model.py et config.py divergent sur ces valeurs."""
    return _load_trained_model(
        checkpoint_path=Config.MODEL_WEIGHTS_PATH,
        num_classes=Config.NUM_CLASSES,
        device=Config.DEVICE,
    )


def postprocess_prediction(pred_mask, num_classes=None, min_area_ratio=None, fill_holes=None):
    """Ne garde que la/les composante(s) connexe(s) dominante(s) par poumon et remplit
    les trous internes -- élimine les fragments isolés (faux positifs)."""
    num_classes = Config.NUM_CLASSES if num_classes is None else num_classes
    min_area_ratio = Config.MIN_AREA_RATIO if min_area_ratio is None else min_area_ratio
    fill_holes = Config.FILL_HOLES if fill_holes is None else fill_holes

    cleaned = np.zeros_like(pred_mask)
    for c in range(1, num_classes):
        class_mask = (pred_mask == c).astype(np.uint8)
        if class_mask.sum() == 0:
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        if num_labels <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = areas.max()
        keep_labels = [i + 1 for i, a in enumerate(areas) if a >= min_area_ratio * largest_area]
        keep_mask = np.isin(labels, keep_labels)
        if fill_holes:
            keep_mask = ndi.binary_fill_holes(keep_mask)
        cleaned[keep_mask] = c
    return cleaned


def draw_lung_mask(img_gray, mask, right_color=(255, 0, 0), left_color=(0, 255, 0), alpha=0.4):
    """Superpose les masques poumon droit/gauche sur l'image grayscale (fond semi-transparent)."""
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    overlay = img_rgb.copy()
    overlay[mask == 1] = right_color
    overlay[mask == 2] = left_color
    blended = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    return blended.astype(np.uint8)


def _array_to_base64_png(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict(model, image_bytes: bytes, apply_postprocessing=None):
    """Pipeline complet : bytes upload -> préprocessing -> modèle -> post-traitement (si activé)."""
    apply_postprocessing = Config.APPLY_POSTPROCESSING if apply_postprocessing is None else apply_postprocessing

    img_norm, img_display = preprocess_from_bytes(image_bytes)
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float().to(Config.DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        mean_confidence = probs.max(dim=1).values.mean().item()

    if apply_postprocessing:
        pred_mask = postprocess_prediction(pred_mask)

    return pred_mask, img_display, mean_confidence


def lung_pixel_stats(pred_mask):
    """Statistiques par classe -- pixels et ratio, format attendu par l'UI Streamlit."""
    total_pixels = pred_mask.size
    stats = {}
    for i, name in enumerate(Config.CLASS_NAMES):
        pixels = int(np.sum(pred_mask == i))
        stats[name] = {"pixels": pixels, "ratio": round(pixels / total_pixels, 4)}
    return stats


def run_inference(model, image_bytes: bytes, apply_postprocessing: bool = True) -> dict:
    """Point d'entrée utilisé par app.py -- renvoie exactement les champs de PredictResponse."""
    pred_mask, img_display, mean_confidence = predict(model, image_bytes, apply_postprocessing)

    overlay = draw_lung_mask(img_display, pred_mask)
    classes_present = sorted(int(c) for c in np.unique(pred_mask))
    class_areas = lung_pixel_stats(pred_mask)
    mask_display = (pred_mask * (255 // max(Config.NUM_CLASSES - 1, 1))).astype(np.uint8)

    return {
        "mask_png_base64": _array_to_base64_png(mask_display),
        "overlay_png_base64": _array_to_base64_png(overlay),
        "classes_present": classes_present,
        "class_areas": class_areas,
        "mean_confidence": round(mean_confidence, 4),
    }