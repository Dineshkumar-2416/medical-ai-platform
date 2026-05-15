import torch
from pathlib import Path


class EarlyStopping:
    """
    Stop training when monitored metric stops improving.

    Prevents overfitting — if val loss hasn't improved in `patience`
    epochs, training is stopped. Standard in all production training pipelines.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (
            value < self.best_value - self.min_delta if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """
    Save model when monitored metric improves.
    Keeps only the single best checkpoint (saves disk space).
    """

    def __init__(self, checkpoint_dir: str, monitor: str = "val_auc", mode: str = "max"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor    = monitor
        self.mode       = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_path  = None

    def step(self, model: torch.nn.Module, value: float, epoch: int, metadata: dict = None) -> bool:
        improved = (
            value > self.best_value if self.mode == "max"
            else value < self.best_value
        )
        if improved:
            self.best_value = value
            save_path = self.checkpoint_dir / f"best_model_epoch{epoch:03d}_{self.monitor}{value:.4f}.pth"

            # Remove previous best to avoid accumulating checkpoints
            if self.best_path and self.best_path.exists():
                self.best_path.unlink()

            # Convert any numpy scalars to Python native types for weights_only safety
            clean_meta = {k: float(v) if hasattr(v, 'item') else v
                          for k, v in (metadata or {}).items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_metric": float(value),
                "metadata": clean_meta,
            }, save_path)

            self.best_path = save_path
            return True
        return False
