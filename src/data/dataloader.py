from torch.utils.data import DataLoader, random_split

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transforms, get_val_transforms


def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, ChestXrayDataset]:
    """
    Build train, val, and test DataLoaders.

    The official val split has only 16 images — too small for meaningful
    validation. So we ignore it and carve 20% out of train ourselves.

    Returns:
        train_loader, val_loader, test_loader, train_dataset
        (train_dataset returned so caller can access class_weights)
    """
    import torch

    train_dataset = ChestXrayDataset(
        root_dir=f"{data_dir}/train",
        transform=get_train_transforms(image_size),
    )
    test_dataset = ChestXrayDataset(
        root_dir=f"{data_dir}/test",
        transform=get_val_transforms(image_size),
    )

    # Split train → train + val
    n_total = len(train_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split_ds = random_split(train_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_split_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset
