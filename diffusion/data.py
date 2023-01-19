""" Module for loading the MNIST dataset"""
from typing import Tuple
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset
from torch.utils.data import ConcatDataset
import numpy as np
from pathlib import Path
from typing import List

def get_transformer() -> transforms.Compose:
    """ Transformer for transforming the MNIST dataset"""
    data_transforms: List = [
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    return transforms.Compose(data_transforms)

def load_mnist_dataset(path: Path, download: bool = False) -> Tuple[VisionDataset, VisionDataset]:
    """ Loads the MNIST dataset and transforms in into the correct format"""
    path_str = path.as_posix()

    # Process train and test dataset:
    data_transform = get_transformer()
    train = datasets.MNIST(root=path_str, train=True, transform=data_transform, download=download)
    test = datasets.MNIST(root=path_str, train=False, transform=data_transform, download=download)
    return train, test