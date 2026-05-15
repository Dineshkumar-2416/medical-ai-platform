"""
Entry point for training the chest X-ray pneumonia detection model.

Usage:
    python train.py

All hyperparameters are in configs/config.yaml — never hardcode values here.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn

os.environ["TORCH_HOME"] = "D:/torch-cache"
os.environ["MLFLOW_TRACKING_URI"] = "file:///D:/Personal Projects/Project_2/mlruns"
sys.path.insert(0, ".")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg    = load_config()
    t_cfg  = cfg["training"]
    device = t_cfg["device"] if torch.cuda.is_available() else "cpu"

    print(f"Device       : {device}")
    print(f"Model        : {t_cfg['model']}")
    print(f"Batch size   : {t_cfg['batch_size']}")
    print(f"Epochs       : {t_cfg['epochs']}")
    print(f"Learning rate: {t_cfg['learning_rate']}")
    print()

    # ── 1. Data ───────────────────────────────────────────────────────
    from src.data.dataloader import get_dataloaders
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        data_dir    = cfg["data"]["raw_dir"] + "/chest_xray",
        image_size  = cfg["data"]["image_size"],
        batch_size  = t_cfg["batch_size"],
        num_workers = t_cfg["num_workers"],
        val_split   = t_cfg["val_split"],
        seed        = t_cfg["seed"],
    )

    # ── 2. Model ──────────────────────────────────────────────────────
    from src.models.resnet import ChestXrayModel
    model = ChestXrayModel(
        num_classes   = cfg["data"]["num_classes"],
        dropout       = cfg["model"]["resnet18"]["dropout"],
        freeze_layers = cfg["model"]["resnet18"]["freeze_layers"],
    ).to(device)
    model.summary()
    print()

    # ── 3. Loss — weighted for class imbalance ────────────────────────
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: NORMAL={class_weights[0]:.3f}  PNEUMONIA={class_weights[1]:.3f}")

    # ── 4. Optimizer ──────────────────────────────────────────────────
    # AdamW = Adam + proper weight decay decoupled from gradient update
    # Only pass trainable parameters — frozen ones don't need an optimizer slot
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = t_cfg["learning_rate"],
        weight_decay = t_cfg["weight_decay"],
    )

    # ── 5. Scheduler — reduce LR when val loss plateaus ───────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # ── 6. Train ──────────────────────────────────────────────────────
    from src.training.trainer import Trainer
    trainer = Trainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        optimizer       = optimizer,
        scheduler       = scheduler,
        criterion       = criterion,
        device          = device,
        checkpoint_dir  = t_cfg["checkpoint_dir"],
        early_stopping_patience = t_cfg["early_stopping_patience"],
        mixed_precision = t_cfg["mixed_precision"],
    )

    best_model_path = trainer.fit(
        epochs          = t_cfg["epochs"],
        experiment_name = cfg["mlflow"]["experiment_name"],
    )

    # ── 7. Evaluate on test set ───────────────────────────────────────
    print("Evaluating best model on test set...")
    from src.training.metrics import compute_metrics, format_metrics

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_metrics = compute_metrics(all_labels, all_probs)
    print(f"\nTest Set Results:")
    print(f"  {format_metrics(test_metrics)}")
    print(f"  TP={test_metrics['tp']}  FP={test_metrics['fp']}  TN={test_metrics['tn']}  FN={test_metrics['fn']}")
    print(f"\nDone. Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
