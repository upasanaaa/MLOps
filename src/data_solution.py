from __future__ import annotations
import torch

DATA_PATH = "data/corruptmnist"


def safe_torch_load(filepath: str) -> torch.Tensor:
    """Safely load a torch tensor to avoid FutureWarnings."""
    try:
        return torch.load(filepath, weights_only=True)
    except TypeError:
        return torch.load(filepath)


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []

    # Load and concatenate training data
    for i in range(6):
        train_images.append(safe_torch_load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(safe_torch_load(f"{DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load test data
    test_images: torch.Tensor = safe_torch_load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = safe_torch_load(f"{DATA_PATH}/test_target.pt")

    # Reshape and convert data types
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Create datasets
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set
