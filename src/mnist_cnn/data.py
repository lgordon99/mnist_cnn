from pathlib import Path
import torch
import typer
# from torch.utils.data import Dataset

# class MyDataset(Dataset):
#     """My custom dataset."""

#     def __init__(self, raw_data_path: Path) -> None:
#         self.data_path = raw_data_path

#     def __len__(self) -> int:
#         """Return the length of the dataset."""

#     def __getitem__(self, index: int):
#         """Return a given sample from the dataset."""

#     def preprocess(self, processed_dir: Path) -> None:
#         """Preprocess the raw data and save it to the output folder."""

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

def preprocess(raw_dir: Path, processed_dir: Path) -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")

def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess)
