import os.path
from typing import Callable, Dict, List

import datasets
import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import CenterCrop, transforms

from ..env import format_with_env
# photo_augmentations.py

import torch
from typing import List

from torchvision.transforms import transforms


class PhotometryTransform:
    """
    A callable class that applies a series of preprocessing steps to a single
    15-dimensional photometry tensor.

    The input tensor order is assumed to be fixed:
    - 5 psfMag (u,g,r,i,z)
    - 5 extinction (u,g,r,i,z)
    - 5 psfMagErr (u,g,r,i,z)

    The processing steps are:
    1. Feature Engineering:
        a. Perform extinction correction (corrected_mag = psfMag - extinction).
        b. Create 4 color features from corrected magnitudes.
    2. Log Transformation: Apply log transform to specified features (errors).
    3. Standardization: Apply Z-score normalization using pre-computed stats.

    The output is a fully preprocessed 19-dimensional tensor.
    """

    def __init__(self,
                 log_transform_indices: List[int] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                 epsilon: float = 1e-6):
        """
        Initializes the photometry transformation pipeline.

        Args:
            mean (torch.Tensor): A tensor of shape (19,) containing the pre-computed means
                                 for the FINAL 19-dimensional feature vector.
            std (torch.Tensor): A tensor of shape (19,) containing the pre-computed standard
                                deviations for the FINAL 19-dimensional feature vector.
            log_transform_indices (List[int]): A list of indices in the FINAL 19-dim vector
                                               that should be log-transformed. Defaults to
                                               error features.
            epsilon (float): A small value to add before log transform to avoid log(0).
        """
        self.mean = torch.tensor([21.4684, 20.9255, 20.7705, 20.6578, 20.5311, -2.1243, -2.3738, -2.7422,
                                  -3.0391, -3.3351, -2.0534, -3.2229, -3.1252, -2.9056, -1.8159, 0.5431,
                                  0.1548, 0.1130, 0.1263])
        self.std = torch.tensor([0.9903, 0.5818, 0.6110, 0.6164, 0.6434, 0.6759, 0.6759, 0.6759, 0.6759,
                                 0.6759, 0.7275, 0.3790, 0.4720, 0.5170, 0.5869, 0.7597, 0.2337, 0.1863,
                                 0.2952])
        self.log_indices = log_transform_indices
        self.epsilon = epsilon

        # --- Define indices for clarity based on the 15-dim input tensor ---
        self.mag_indices = slice(0, 5)
        self.ext_indices = slice(5, 10)
        self.err_indices = slice(10, 15)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the full transformation pipeline to a BATCH of 15-dim tensors.
        Args:
            batch (torch.Tensor): A tensor of shape (batch_size, 15).
        Returns:
            torch.Tensor: A preprocessed tensor of shape (batch_size, 19).
        """
        # All operations are already vectorized, so they work on batches!
        corrected_mags = batch[:, self.mag_indices] - batch[:, self.ext_indices]
        colors = corrected_mags[:, :-1] - corrected_mags[:, 1:]

        processed_vector = torch.cat([
            corrected_mags,
            batch[:, self.ext_indices],
            batch[:, self.err_indices],
            colors
        ], dim=1)  # Use dim=1 for batch concatenation

        if self.log_indices:
            # Slicing works on batches too
            processed_vector[:, self.log_indices] = torch.log(processed_vector[:, self.log_indices] + self.epsilon)

        standardized_vector = (processed_vector - self.mean.to(batch.device)) / (
                self.std.to(batch.device) + self.epsilon)

        return standardized_vector


class AstroClipDataloader(L.LightningDataModule):
    def __init__(
            self,
            path: str,
            columns: List[str] = ["image", "spectrum", "params"],
            batch_size: int = 512,
            num_workers: int = 10,
            collate_fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Check if hparams are nested under 'init_args' (training scenario)
        if 'init_args' in self.hparams:
            self.hparams_ = self.hparams.init_args
        # Otherwise, hparams are flat (inference scenario)
        else:
            self.hparams_ = self.hparams

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        # 1. 从 hparams 获取原始的、可能未被替换的路径
        raw_path = self.hparams_['path']

        # 2. 手动调用 format_with_env 函数来确保环境变量被正确替换
        expanded_path = format_with_env(raw_path)

        # 3. 使用被替换后的完整路径来加载数据集
        train_path = os.path.join(expanded_path, 'train_dataset')
        test_path = os.path.join(expanded_path, 'test_dataset')
        self.train_dataset = datasets.load_from_disk(train_path)
        self.test_dataset = datasets.load_from_disk(test_path)

        self.train_dataset.set_format(type="torch", columns=self.hparams_['columns'])
        self.test_dataset.set_format(type="torch", columns=self.hparams_['columns'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams_['batch_size'],
            shuffle=True,
            num_workers=self.hparams_['num_workers'],  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.hparams_['collate_fn'] if 'collate_fn' in self.hparams_ else None
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams_['batch_size'],
            num_workers=self.hparams_['num_workers'],  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.hparams_['collate_fn'] if 'collate_fn' in self.hparams_ else None
        )


class AstroClipCollator:
    def __init__(
            self,
            center_crop: int = 64,
    ):
        self.center_crop = CenterCrop(center_crop)

        self.image_trans = transforms.Normalize(mean=[0.003918, 0.00564221, 0.00870101, 0.01226911, 0.01992336],
                                                std=[0.4310049, 0.27417892, 0.38182727, 0.47646362, 1.3710657])

        self.photometry_transform = PhotometryTransform()

    def _process_images(self, images):
        images = self.image_trans(self.center_crop(images))
        return images

    def _process_photometry(self, photometry):
        photometry = self.photometry_transform(photometry)
        return photometry

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)
        # process images
        samples["image"] = self._process_images(samples["image"])

        # MODIFICATION: Process photometry data for the entire batch
        processed_params = self._process_photometry(samples["params"])
        samples["params"] = processed_params

        return samples
