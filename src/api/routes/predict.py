import io
import base64
import logging
import numpy as np
from PIL import Image
import cv2

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from src.api.schemas import PredictionResponse

logger    = logging.getLogger(__name__)
router    = APIRouter()

ALLOWED_TYPES     = {"image/jpeg", "image/png", "image/jpg"}
MAX_SIZE_BYTES    = 10 * 1024 * 1024    # 10 MB
LOW_CONF_THRESHOLD = 0.60               # warn if confidence below this


def _encode_image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy RGB image → PNG bytes → base64 string for JSON transport."""
    img_bgr   = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict pneumonia from a chest X-ray image.

    Accepts: JPEG or PNG, max 10MB
    Returns: prediction, confidence, GradCAM overlay as base64 PNG
    """
    # ── Validate file type ────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG."
        )

    # ── Read + validate size ──────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents)/1e6:.1f}MB. Max is 10MB."
        )

    # ── Validate it's a real image ────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="File is not a valid image.")

    # ── Run inference + GradCAM ───────────────────────────────────────
    try:
        model       = request.app.state.model
        device      = request.app.state.device
        image_size  = request.app.state.image_size

        from src.explainability.gradcam import run_gradcam

        # Save image to a temp buffer for run_gradcam (expects a file path or array)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as tmp:
            image.save(tmp.name, format="JPEG")
            tmp_path = tmp.name

        result = run_gradcam(model, tmp_path, device=device, image_size=image_size)
        os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # ── Build response ────────────────────────────────────────────────
    gradcam_b64 = _encode_image_to_base64(result["overlay"])
    low_conf    = result["confidence"] < LOW_CONF_THRESHOLD
    message     = (
        f"Low confidence ({result['confidence']*100:.1f}%). "
        "Manual review recommended."
        if low_conf else None
    )

    logger.info(
        f"Prediction: {result['prediction']} | "
        f"Confidence: {result['confidence']:.4f} | "
        f"File: {file.filename} | "
        f"Low confidence: {low_conf}"
    )

    return PredictionResponse(
        prediction      = result["prediction"],
        confidence      = result["confidence"],
        prob_normal     = result["prob_normal"],
        prob_pneumonia  = result["prob_pneumonia"],
        gradcam_image   = gradcam_b64,
        low_confidence  = low_conf,
        message         = message,
    )
