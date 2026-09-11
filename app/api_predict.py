"""
FastAPI app serving the U-Net++ lung segmentation model.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> liveness check + model/device info
    POST /predict  -> multipart file upload (chest X-ray) -> mask + overlay (base64 PNG)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from model_utils import load_model, run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lung-segmentation-api")

ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg",
    "application/dicom", "application/octet-stream",  # DICOM often arrives untyped
}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}

# Holds the loaded model; populated at startup, avoids reloading weights per request.
ml_state = {"model": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_state["model"] = load_model()
        logger.info(f"Model loaded successfully on {Config.DEVICE} "
                    f"(backbone={getattr(Config, 'BEST_BACKBONE', 'unetpp_resnet34')}, "
                    f"postprocessing_default={Config.APPLY_POSTPROCESSING})")
    except Exception as e:
        # Don't crash the whole process on a bad checkpoint/config -- start the app so
        # /health can report the failure instead of the container just dying silently.
        ml_state["load_error"] = str(e)
        logger.exception("Failed to load model at startup")
    yield
    ml_state.clear()


app = FastAPI(title="Lung Segmentation API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin before going to production
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    device: str
    img_size: int
    num_classes: int
    class_names: list[str]
    model_loaded: bool
    load_error: str | None = None


class PredictResponse(BaseModel):
    mask_png_base64: str
    overlay_png_base64: str
    classes_present: list[int]
    class_areas: dict
    mean_confidence: float


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if ml_state["model"] is not None else "degraded",
        device=str(Config.DEVICE),
        img_size=Config.IMG_SIZE,
        num_classes=Config.NUM_CLASSES,
        class_names=Config.CLASS_NAMES,
        model_loaded=ml_state["model"] is not None,
        load_error=ml_state["load_error"],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), postprocess: bool = Config.APPLY_POSTPROCESSING):
    if ml_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {ml_state['load_error'] or 'unknown error'}",
        )

    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if file.content_type not in ALLOWED_CONTENT_TYPES and file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file (content type: {file.content_type}, extension: {file_ext or 'none'})",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = run_inference(ml_state["model"], image_bytes, apply_postprocessing=postprocess)
    except ValueError as e:
        # Bad/corrupt image, unreadable format, etc. -- client error.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Model forward pass failure, shape mismatch, OOM, etc. -- server error, don't
        # leak the raw stack trace to the client.
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed — see server logs.")

    return PredictResponse(**result)