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


# --- 新增/修改：Batch-friendly PhotometryTransform ---
class PhotometryTransform:
    """
    Preprocessing pipeline for SDSS Photometry Data (Batch Version).
    Transforms raw (B, 15) input into a sequence-ready (B, 5, 2) tensor.

    Input Shape: (Batch, 15)
    - 0-4:  psfMag (u, g, r, i, z)
    - 5-9:  extinction (u, g, r, i, z)
    - 10-14: psfMagErr (u, g, r, i, z)

    Output Shape: (Batch, 5, 2)
    - Dim 1: Sequence (u, g, r, i, z)
    - Dim 2: Features [Corrected_Magnitude, Log_Magnitude_Error]
    """

    def __init__(self, epsilon: float = 1e-6):
        # =========================================================================
        # TODO: 请使用 calculate_stats.py 计算出的真实 Mean 和 Std 替换此处！
        # 这里的形状必须是 (5, 2)
        # =========================================================================
        self.mean = torch.tensor(
            [[21.4684, -2.0534],
             [20.9255, -3.2229],
             [20.7705, -3.1252],
             [20.6578, -2.9056],
             [20.5311, -1.8159]])
        self.std = torch.tensor(
            [[0.9903, 0.7275],
             [0.5818, 0.3790],
             [0.6110, 0.4720],
             [0.6164, 0.5170],
             [0.6434, 0.5869]]
        )
        self.epsilon = epsilon

        # Indices slicing (applies to the last dimension)
        self.mag_indices = slice(0, 5)
        self.ext_indices = slice(5, 10)
        self.err_indices = slice(10, 15)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to a BATCH of tensors.

        Args:
            batch (torch.Tensor): Shape (Batch_Size, 15)

        Returns:
            torch.Tensor: Shape (Batch_Size, 5, 2)
        """
        # Ensure input is 2D (Batch, 15)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)

        device = batch.device

        # 1. Physics Correction: Intrinsic Mag = psfMag - extinction
        # batch[:, slice] operates on the last dimension
        raw_mags = batch[:, self.mag_indices]
        extinctions = batch[:, self.ext_indices]
        raw_errs = batch[:, self.err_indices]

        corrected_mags = raw_mags - extinctions

        # 2. Log Transformation for Errors
        log_errs = torch.log(raw_errs + self.epsilon)

        # 3. Reshape / Stacking
        # We want (Batch, 5, 2).
        # corrected_mags is (B, 5), log_errs is (B, 5)
        # stack(dim=2) results in (B, 5, 2) where [:, i, :] is [mag_i, err_i]
        output_tensor = torch.stack([corrected_mags, log_errs], dim=2)

        # 4. Standardization (Z-score)
        # Broadcast mechanism: (B, 5, 2) - (5, 2) works automatically in PyTorch
        standardized_tensor = (output_tensor - self.mean.to(device)) / (self.std.to(device) + self.epsilon)

        return standardized_tensor

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
            collate_fn=self.hparams_['collate_fn'] if 'collate_fn' in self.hparams_ else None,
            # --- 新增优化参数 ---
            # pin_memory=True,  # 必须开启！加速 CPU -> GPU 传输
            # persistent_workers=True  # 必须开启！避免每个Epoch重建进程
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams_['batch_size'],
            num_workers=self.hparams_['num_workers'],  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.hparams_['collate_fn'] if 'collate_fn' in self.hparams_ else None,
            # --- 新增优化参数 ---
            # pin_memory=True,  # 必须开启！加速 CPU -> GPU 传输
            # persistent_workers=True  # 必须开启！避免每个Epoch重建进程
        )


class AstroClipCollator:
    def __init__(
            self,
            center_crop: int = 64,
    ):
        self.center_crop = CenterCrop(center_crop)

        self.image_trans = transforms.Normalize(mean=[0.003918, 0.00564221, 0.00870101, 0.01226911, 0.01992336],
                                                std=[0.4310049, 0.27417892, 0.38182727, 0.47646362, 1.3710657])

        # Initialize the Batch-friendly transform
        self.photometry_transform = PhotometryTransform()

    def _process_images(self, images):
        images = self.image_trans(self.center_crop(images))
        return images

    def _process_photometry(self, photometry):
        # photometry is expected to be a Batch Tensor (B, 15) here
        photometry = self.photometry_transform(photometry)
        return photometry

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)
        # process images
        samples["image"] = self._process_images(samples["image"])

        samples["params"] = self._process_photometry(samples["params"])
        return samples
