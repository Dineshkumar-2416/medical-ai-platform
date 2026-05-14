import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from src.training.metrics import compute_metrics, format_metrics
from src.training.callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """
    Production training loop for chest X-ray classification.

    Features:
      - Mixed precision training (AMP) for RTX 3050 VRAM efficiency
      - MLflow experiment tracking (every metric, every epoch)
      - Early stopping + best model checkpointing
      - Separate train / val loops with correct model modes
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: str,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 5,
        mixed_precision: bool = True,
    ):
        self.model       = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.criterion   = criterion
        self.device      = device
        self.mixed_precision = mixed_precision

        self.scaler = GradScaler("cuda") if mixed_precision else None

        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")
        self.checkpoint     = ModelCheckpoint(checkpoint_dir, monitor="val_auc", mode="max")

    # ------------------------------------------------------------------
    def _train_epoch(self) -> tuple[float, list, list]:
        self.model.train()   # activates dropout + batch norm training mode

        total_loss = 0.0
        all_labels, all_probs = [], []

        for images, labels in tqdm(self.train_loader, desc="  Train", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with autocast("cuda"):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss   = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()

            probs = torch.softmax(logits.detach(), dim=1)[:, 1]  # P(PNEUMONIA)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, all_labels, all_probs

    # ------------------------------------------------------------------
    def _val_epoch(self) -> tuple[float, list, list]:
        self.model.eval()   # disables dropout, uses running BN statistics

        total_loss = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="  Val  ", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.mixed_precision:
                    with autocast("cuda"):
                        logits = self.model(images)
                        loss   = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, all_labels, all_probs

    # ------------------------------------------------------------------
    def fit(self, epochs: int, experiment_name: str = "chest-xray-pneumonia"):

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log hyperparameters once
            mlflow.log_params({
                "epochs":           epochs,
                "optimizer":        type(self.optimizer).__name__,
                "lr":               self.optimizer.param_groups[0]["lr"],
                "mixed_precision":  self.mixed_precision,
                "train_batches":    len(self.train_loader),
                "val_batches":      len(self.val_loader),
            })

            print(f"\n{'='*60}")
            print(f"Training for {epochs} epochs on {self.device.upper()}")
            print(f"{'='*60}\n")

            for epoch in range(1, epochs + 1):
                print(f"Epoch {epoch:>3}/{epochs}")

                # --- Train ---
                train_loss, train_labels, train_probs = self._train_epoch()
                train_metrics = compute_metrics(train_labels, train_probs)

                # --- Validate ---
                val_loss, val_labels, val_probs = self._val_epoch()
                val_metrics = compute_metrics(val_labels, val_probs)

                # --- Scheduler step (based on val loss) ---
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]

                # --- Log to MLflow ---
                mlflow.log_metrics({
                    "train_loss":        train_loss,
                    "train_auc":         train_metrics["auc"],
                    "train_sensitivity": train_metrics["sensitivity"],
                    "train_specificity": train_metrics["specificity"],
                    "val_loss":          val_loss,
                    "val_auc":           val_metrics["auc"],
                    "val_sensitivity":   val_metrics["sensitivity"],
                    "val_specificity":   val_metrics["specificity"],
                    "val_f1":            val_metrics["f1"],
                    "learning_rate":     current_lr,
                }, step=epoch)

                # --- Print epoch summary ---
                print(f"  Train loss: {train_loss:.4f} | {format_metrics(train_metrics)}")
                print(f"  Val   loss: {val_loss:.4f} | {format_metrics(val_metrics)}")
                print(f"  LR: {current_lr:.2e}")

                # --- Checkpoint best model ---
                saved = self.checkpoint.step(
                    self.model, val_metrics["auc"], epoch,
                    metadata={**val_metrics, "epoch": epoch},
                )
                if saved:
                    print(f"  Saved best model (val_auc={val_metrics['auc']:.4f})")
                    mlflow.log_artifact(str(self.checkpoint.best_path))

                # --- Early stopping ---
                if self.early_stopping.step(val_loss):
                    print(f"\n  Early stopping at epoch {epoch} (no improvement for {self.early_stopping.patience} epochs)")
                    break

                print()

            print(f"\n{'='*60}")
            print(f"Training complete.")
            print(f"Best val AUC : {self.checkpoint.best_value:.4f}")
            print(f"Best model   : {self.checkpoint.best_path}")
            print(f"{'='*60}\n")

            if self.checkpoint.best_path:
                mlflow.log_metric("best_val_auc", self.checkpoint.best_value)

        return self.checkpoint.best_path
