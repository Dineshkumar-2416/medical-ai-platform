"""
Central place to load and save model checkpoints.

In production, models are never loaded from arbitrary paths — always
through a registry that enforces versioning and metadata.
"""
import torch
from pathlib import Path
from src.models.resnet import ChestXrayModel


def build_model(
    num_classes: int = 2,
    dropout: float = 0.3,
    freeze_layers: int = 6,
    device: str = "cuda",
) -> ChestXrayModel:
    model = ChestXrayModel(num_classes=num_classes, dropout=dropout, freeze_layers=freeze_layers)
    model = model.to(device)
    return model


def save_checkpoint(model: ChestXrayModel, path: str, metadata: dict = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, device: str = "cuda", **model_kwargs) -> ChestXrayModel:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(device=device, **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    meta = checkpoint.get("metadata", {})
    if meta:
        print(f"Loaded checkpoint metadata: {meta}")
    return model
