# Dataset file for DESI Legacy Survey data
import logging
import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image as im
from datasets import load_from_disk
from torchvision.datasets import VisionDataset

logger = logging.getLogger("astrodino")

class QuasarDataset(VisionDataset):
    def __init__(
            self,
            *,
            split: str,  # 数据集分割类型（训练/验证/测试）
            root: str,  # 数据集根目录路径
            extra: str = None,  # 额外数据路径（可选）
            transforms: Optional[Callable] = None,  # 图像变换函数（可选）
            transform: Optional[Callable] = None,  # 单例图像变换（可选）
            target_transform: Optional[Callable] = None,  # 目标变换函数（可选）
    ) -> None:
        # 调用父类构造函数初始化
        super().__init__(root, transforms, transform, target_transform)
        self._split = split  # 保存数据集分割类型
        self.root  = root

        if self._split=="train":
            self.root = os.path.join(root, "train_dataset")
        elif self._split=="test":
            self.root = os.path.join(root, "test_dataset")
        self.data= load_from_disk(self.root)
        self.data.set_format(type="torch", columns=["image"])


    @property
    def split(self):
        return self._split  # 返回当前数据集分割类型

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        # 1. 加载原始数据
        img = self.data['image'][idx]

        target = None  # 当前无目标标签

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img,target
