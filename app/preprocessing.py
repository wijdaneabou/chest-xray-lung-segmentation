import io

import cv2
import numpy as np
import pydicom

from config import Config


def load_gray_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a grayscale array. Tries a standard image codec first
    (PNG/JPEG/...); if that fails, falls back to DICOM -- so the API stays robust
    even if a client sends a .dcm file directly, bypassing the Streamlit UI."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img

    try:
        ds = pydicom.dcmread(io.BytesIO(image_bytes))
        img = ds.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
        return img.astype(np.uint8)
    except Exception as e:
        raise ValueError("Could not decode image — unsupported or corrupt file.") from e


def percentile_normalize(img: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return img.astype(np.uint8)
    img_norm = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
    return img_norm.astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def prepare_image_array(img_raw: np.ndarray) -> np.ndarray:
    """Pipeline déterministe identique à l'entraînement : percentile -> CLAHE -> resize."""
    img = percentile_normalize(img_raw)
    img = apply_clahe(img)
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return img


def zscore_normalize(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32) / 255.0
    img_norm = (img_float - Config.NORM_MEAN) / (Config.NORM_STD + 1e-8)
    return img_norm[np.newaxis, :, :]


def preprocess_from_bytes(image_bytes: bytes):
    """image_bytes -> (tenseur modèle (1, H, W), image prétraitée pour affichage)."""
    img_raw = load_gray_from_bytes(image_bytes)
    img_resized = prepare_image_array(img_raw)
    img_norm = zscore_normalize(img_resized)
    return img_norm, img_resized