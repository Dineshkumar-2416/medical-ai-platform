import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray binary classification.

    Folder structure expected:
        root/
          NORMAL/     *.jpeg
          PNEUMONIA/  *.jpeg
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir : path to split folder (e.g. data/raw/chest_xray/train)
            transform: albumentations Compose pipeline (or None)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []   # list of (image_path, label)
        self.class_counts = {0: 0, 1: 0}

        self._load_samples()

    def _load_samples(self):
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected class folder not found: {class_dir}")

            images = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            for img_path in sorted(images):
                self.samples.append((img_path, label))
                self.class_counts[label] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Open as RGB — X-rays are grayscale but models expect 3-channel input
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            import numpy as np
            image = self.transform(image=np.array(image))["image"]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights to handle class imbalance.

        Used as weight= argument in nn.CrossEntropyLoss.
        More frequent class gets lower weight, rarer class gets higher weight.
        """
        total = sum(self.class_counts.values())
        weights = [total / (2 * self.class_counts[i]) for i in range(2)]
        return torch.tensor(weights, dtype=torch.float32)

    def summary(self):
        print(f"Dataset: {self.root_dir.name}")
        print(f"  Total images : {len(self.samples)}")
        print(f"  NORMAL       : {self.class_counts[0]}")
        print(f"  PNEUMONIA    : {self.class_counts[1]}")
        ratio = self.class_counts[1] / max(self.class_counts[0], 1)
        print(f"  Imbalance    : {ratio:.2f}x more PNEUMONIA than NORMAL")
        weights = self.get_class_weights()
        print(f"  Class weights: NORMAL={weights[0]:.3f}, PNEUMONIA={weights[1]:.3f}")
