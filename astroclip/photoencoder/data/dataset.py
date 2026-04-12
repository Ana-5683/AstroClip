# dataset.py
import os.path

import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Optional, List, Callable

from .photo_augmentations import PhotometryTransform


class PhotometryDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing photometry data from a file.
    """

    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None):
        """
        Initializes the dataset.

        Args:
            data_path (str): Path to the data file.
                             The file should contain columns with the 15 feature names.
            transform (Optional[Callable]): An instance of the PhotometryTransform
                                                       class to apply to each sample.
        """
        # Load the entire dataset into memory using pandas.
        # For very large datasets, consider using memory-efficient formats like Parquet.
        if split=='train':
            data_path=os.path.join(data_path, 'train_dataset')
        elif split=='test':
            data_path=os.path.join(data_path, 'test_dataset')

        self.dataset=datasets.load_from_disk(data_path)
        self.transform = transform

        # It's good practice to convert the DataFrame to a list of dictionaries
        # for faster access during training, as df.iloc can be slow.
        self.dataset.set_format(type="torch", columns=['params'])
        print(f"Loaded {len(self.dataset)} samples from {data_path}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves one sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The preprocessed 19-dimensional feature tensor.
        """
        # Get the sample as a dictionary
        sample = self.dataset["params"][idx]

        # Apply the transformations if they are provided
        if self.transform:
            sample = self.transform(sample)
        else:
            # If no transform, convert to tensor manually (not recommended for training)
            sample = torch.tensor(list(sample.values()), dtype=torch.float32)

        return sample
