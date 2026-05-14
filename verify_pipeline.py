"""
Run this script to verify the entire Phase 2 data pipeline works.
  python verify_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = "data/raw/chest_xray"


def main():
    print("=" * 55)
    print("PHASE 2 — DATA PIPELINE VERIFICATION")
    print("=" * 55)

    # 1. Dataset
    print("\n[1/4] Loading datasets...")
    from src.data.dataset import ChestXrayDataset
    train_ds = ChestXrayDataset(f"{DATA_DIR}/train")
    test_ds  = ChestXrayDataset(f"{DATA_DIR}/test")
    train_ds.summary()
    print()
    test_ds.summary()

    # 2. Class weights
    print("\n[2/4] Class weights (for imbalance handling)...")
    weights = train_ds.get_class_weights()
    print(f"  NORMAL={weights[0]:.4f}  PNEUMONIA={weights[1]:.4f}")
    print("  (Higher weight = model penalized more for missing that class)")

    # 3. Transforms
    print("\n[3/4] Testing transforms...")
    from src.data.transforms import get_train_transforms, get_val_transforms
    import numpy as np
    from PIL import Image

    sample_path = train_ds.samples[0][0]
    img_np = np.array(Image.open(sample_path).convert("RGB"))
    print(f"  Raw image shape : {img_np.shape}")

    train_tfm = get_train_transforms(224)
    tensor = train_tfm(image=img_np)["image"]
    print(f"  After transform : {tensor.shape}  dtype={tensor.dtype}")
    print(f"  Pixel range     : [{tensor.min():.3f}, {tensor.max():.3f}]")

    # 4. DataLoader
    print("\n[4/4] Testing DataLoader (1 batch)...")
    from src.data.dataloader import get_dataloaders
    import torch

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=32,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))
    assert images.shape == torch.Size([32, 3, 224, 224]), "Wrong image shape!"
    assert images.dtype == torch.float32, "Wrong dtype!"
    assert set(labels.numpy().tolist()).issubset({0, 1}), "Wrong labels!"

    print(f"  Batch shape  : {images.shape}  — [batch, channels, height, width]")
    print(f"  Labels shape : {labels.shape}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches  : {len(val_loader)}")
    print(f"  Test batches : {len(test_loader)}")

    print("\n" + "=" * 55)
    print("ALL CHECKS PASSED — Phase 2 complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
