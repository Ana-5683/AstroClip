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


class AstroClipDataloader(L.LightningDataModule):
    def __init__(
            self,
            path: str,
            columns: List[str] = ["image", "spectrum"],
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

        self.trans = transforms.Normalize(mean=[0.003918, 0.00564221, 0.00870101, 0.01226911, 0.01992336],
                                          std=[0.4310049, 0.27417892, 0.38182727, 0.47646362, 1.3710657])

    def _process_images(self, images):
        images = self.trans(self.center_crop(images))
        return images

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)
        # process images
        samples["image"] = self._process_images(samples["image"])
        return samples
