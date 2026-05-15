# ── Stage: Production image ───────────────────────────────────────────────────
# PyTorch official image with CUDA 12.1 + cuDNN 9 (matches our training setup)
# 'runtime' variant — no compiler tools, smaller than 'devel'
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Metadata
LABEL maintainer="medical-ai-platform"
LABEL version="1.0.0"
LABEL description="Chest X-ray Pneumonia Detection API"

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 required by OpenCV (for GradCAM overlay)
# libglib2.0-0 required by OpenCV on Ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy ONLY requirements first — this layer is cached until requirements.txt changes
# Installing deps before copying code = faster rebuilds on code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source code ───────────────────────────────────────────────────
# Copied last — changes frequently, but layer cache already has all deps
COPY src/ ./src/
COPY configs/ ./configs/
# checkpoints/ is mounted as a volume at runtime — not baked into the image

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch-cache

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Docker will call this every 30s — if it fails 3 times, container is "unhealthy"
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --workers 1  : single worker (GPU memory is not shared across workers)
# --host 0.0.0.0 : listen on all interfaces (not just localhost)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
