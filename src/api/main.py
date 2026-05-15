"""
FastAPI application — production inference server.

Startup:  loads model once into GPU memory
Shutdown: releases GPU memory cleanly

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager

os.environ["TORCH_HOME"] = "D:/torch-cache"
sys.path.insert(0, ".")

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.predict import router as predict_router
from src.api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/best_model_epoch018_val_auc0.9951.pth"
IMAGE_SIZE      = 224
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ── Lifespan — runs at startup and shutdown ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...")
    from src.models.resnet import ChestXrayModel

    model = ChestXrayModel().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Store in app.state so every endpoint can access it
    app.state.model      = model
    app.state.device     = DEVICE
    app.state.image_size = IMAGE_SIZE
    app.state.start_time = time.time()
    app.state.model_path = CHECKPOINT_PATH

    logger.info(f"Model loaded. Device={DEVICE}. Ready to serve.")
    yield

    # SHUTDOWN
    logger.info("Shutting down. Releasing GPU memory.")
    del app.state.model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical AI — Chest X-ray Pneumonia Detection",
    description="Production inference API for chest X-ray pneumonia classification with GradCAM explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows browser frontends (Streamlit, React) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(predict_router, tags=["Inference"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    """Returns server health — used by Docker health checks and load balancers."""
    return HealthResponse(
        status       = "healthy",
        model_loaded = hasattr(request.app.state, "model"),
        device       = DEVICE,
        uptime_sec   = round(time.time() - request.app.state.start_time, 2),
        model_path   = CHECKPOINT_PATH,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
