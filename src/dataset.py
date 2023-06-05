"""code adapted from https://github.com/snehjp2/GCNNMorphology/blob/main/src/scripts/dataset.py"""
import sys
import torch
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import Tuple


class Galaxy10DECals(Dataset):
    """Loading Galaxy10 DECals dataset from .h5 file.
    Args:
        dataset_path (string) : path to h5 file
    """

    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = dataset_path
        # self.dataset = None
        self.transform = transform
        with h5py.File(self.dataset_path, "r") as f:
            self.img = f["images"][()]
            self.label = f["ans"][()]
            self.length = len(f["ans"][()])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.img[idx]
        label = torch.tensor(self.label[idx], dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return self.length


if __name__ == "__main__":
    print(sys.argv[1])
    dataset = Galaxy10DECals(sys.argv[1], transform=None)
    # dataset = Galaxy10DECalsTest(sys.argv[1],transform=transform)
    print(len(dataset))
    img, label = dataset[12342]
    print(img.shape)
    print(label)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()
