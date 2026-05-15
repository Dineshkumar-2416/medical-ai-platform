# Medical AI Platform — Chest X-ray Pneumonia Detection

A production-grade medical imaging AI system for binary classification of chest X-rays (Normal vs Pneumonia). Built with a full MLOps stack: transfer learning, REST inference API, GradCAM explainability, containerized deployment, and CI/CD automation.

---

## Results

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | **0.9951** | **0.9620** |
| Sensitivity (Recall) | 0.98 | 0.96 |
| Specificity | 0.97 | 0.94 |
| F1 Score | 0.97 | 0.95 |

Trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset — 5,216 training images across 2 classes with significant class imbalance (NORMAL: PNEUMONIA ≈ 1:3).

---

## Architecture

```
chest-xray/
├── src/
│   ├── data/
│   │   ├── dataset.py          # ChestXrayDataset — class-weighted sampling
│   │   ├── transforms.py       # Albumentations train/val/inference pipelines
│   │   └── dataloader.py       # DataLoader factory with 80/20 train-val split
│   ├── models/
│   │   ├── resnet.py           # ChestXrayModel — ResNet18 backbone via timm
│   │   └── model_registry.py   # Checkpoint save/load utilities
│   ├── training/
│   │   ├── trainer.py          # AMP training loop with MLflow tracking
│   │   ├── metrics.py          # AUC, sensitivity, specificity, F1
│   │   └── callbacks.py        # EarlyStopping, ModelCheckpoint
│   ├── explainability/
│   │   └── gradcam.py          # GradCAM heatmap generation on layer4
│   └── api/
│       ├── main.py             # FastAPI app — lifespan model loading
│       ├── schemas.py          # Pydantic request/response schemas
│       └── routes/predict.py   # POST /predict with file validation
├── docker/
│   ├── Dockerfile              # PyTorch CUDA runtime image
│   ├── Dockerfile.streamlit    # Lightweight dashboard image
│   └── docker-compose.yml      # Multi-service orchestration
├── .github/workflows/
│   └── ci.yml                  # Test + Docker build on every push
├── configs/config.yaml         # All hyperparameters and paths
├── app.py                      # Streamlit dashboard
└── train.py                    # Training entrypoint
```

---

## Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Model | ResNet18 via `timm` | Pretrained ImageNet backbone |
| Training | PyTorch + AMP | Mixed precision, gradient clipping |
| Augmentation | Albumentations | HFlip, rotation, brightness, GaussNoise |
| Explainability | GradCAM | Gradient-weighted class activation maps |
| API | FastAPI + Uvicorn | Async REST inference server |
| Validation | Pydantic v2 | Request/response schema enforcement |
| Dashboard | Streamlit | X-ray upload, prediction, heatmap overlay |
| Experiment Tracking | MLflow | Metrics, params, artifact logging |
| Containerization | Docker + Compose | Reproducible multi-service deployment |
| CI/CD | GitHub Actions | Automated tests + Docker build on push |
| Imbalance Handling | Inverse-frequency class weights | NORMAL=1.945, PNEUMONIA=0.673 |

---

## Model Details

**Backbone:** ResNet18 pretrained on ImageNet (loaded via `timm.create_model`)

**Transfer Learning Strategy:**
- First 6 children (layers) frozen — preserves low-level edge/texture features
- Remaining layers fine-tuned — adapts to medical imaging domain
- Frozen parameters: 683,072 — Trainable: 10,494,466

**Classification Head:**
```
GlobalAvgPool → Dropout(0.3) → Linear(512, 2)
```

**Training Configuration:**
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- Scheduler: ReduceLROnPlateau (`patience=3`, `factor=0.5`)
- Loss: CrossEntropyLoss with class weights
- Mixed Precision: `torch.amp.autocast("cuda")` + `GradScaler("cuda")`
- Gradient Clipping: `max_norm=1.0`
- Early Stopping: `patience=5` on validation loss

---

## GradCAM Explainability

Gradient-weighted Class Activation Mapping highlights the regions in the X-ray that most influenced the model's prediction. Hooks are registered on `model.backbone.layer4` — the deepest convolutional block — to capture spatial gradients before global average pooling.

```python
# Inference pipeline returns
{
    "prediction": "PNEUMONIA",
    "confidence": 0.94,
    "prob_normal": 0.06,
    "prob_pneumonia": 0.94,
    "gradcam_image": "<base64-encoded PNG overlay>"
}
```

---

## Quick Start

### Local (without Docker)

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 3. Start the dashboard (separate terminal)
streamlit run app.py
```

### Docker (recommended)

```bash
# Build images
docker compose -f docker/docker-compose.yml build

# Start all services
docker compose -f docker/docker-compose.yml up
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Server | http://localhost:8000 |
| Swagger UI (API Docs) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## API Reference

### `POST /predict`

Accepts a chest X-ray image (JPEG/PNG, max 10MB) and returns prediction with GradCAM overlay.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.94,
  "prob_normal": 0.06,
  "prob_pneumonia": 0.94,
  "gradcam_image": "<base64 PNG>",
  "low_confidence": false,
  "message": "High confidence prediction"
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "uptime_sec": 142.5,
  "model_path": "checkpoints/best_model_epoch018_val_auc0.9951.pth"
}
```

---

## CI/CD Pipeline

Every push to `main` triggers:

1. **Test job** — installs lightweight deps, runs `pytest tests/`
2. **Build job** — builds the Docker image (runs only if tests pass), uses GitHub Actions layer cache for fast rebuilds

```
push to main
    │
    ├── test (ubuntu-latest, Python 3.11)
    │       pytest tests/ -v
    │
    └── build (needs: test)
            docker build -f docker/Dockerfile
```

---

## Training

```bash
python train.py
```

Logs metrics to MLflow. Start the MLflow UI with:

```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kermany et al., Cell 2018

Place extracted data at:
```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```
