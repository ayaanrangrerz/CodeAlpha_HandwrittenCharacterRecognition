
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=128, data_dir="data"):
    """
    Download MNIST dataset and create training
    and testing DataLoaders.
    """

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    train_dataset = datasets.MNIST(
        root=str(data_path),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(data_path),
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader
