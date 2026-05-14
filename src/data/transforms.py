import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet mean/std — used because ResNet18 was pretrained on ImageNet
# Normalizing our X-rays to the same scale the model "expects"
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Augmentation pipeline for training.

    Each transform is medically justified:
    - HorizontalFlip : X-rays are anatomically symmetric left/right
    - Rotate(±10°)   : Patient positioning variation in real scans
    - Brightness/Contrast : Exposure differences across machines
    - GaussianNoise  : Detector noise simulation
    - NO vertical flip, NO aggressive crops — would destroy anatomy
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Validation/test pipeline — resize + normalize only.
    No augmentation: we need deterministic, reproducible evaluation.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """Identical to val transforms. Used by FastAPI inference endpoint."""
    return get_val_transforms(image_size)
