# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import random
from math import ceil
from typing import List, Tuple
import skimage.filters
import skimage.transform
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from astroclip.astrophoto.data import PhotometryTransform

logger = logging.getLogger("dinov2")


class AsinhTransform(torch.nn.Module):
    """
    对Tensor应用 asinh(x / softening_factor) 变换。
    这是一个强大的对数类变换，可以很好地处理正、负和零值。
    """

    def __init__(self, softening_factor: float = 0.1):
        """
        Args:
            softening_factor (float): 控制线性区到对数区过渡的因子。
                                    较小的值会使变换更快地接近对数行为。
                                    通常需要根据数据的典型值范围进行调整。
        """
        super().__init__()
        self.softening_factor = softening_factor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (torch.Tensor): 输入Tensor。

        Returns:
            torch.Tensor: 变换后的Tensor。
        """
        return torch.asinh(tensor / self.softening_factor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(softening_factor={self.softening_factor})"


class DataAugmentationAstroDINO(object):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=144,
            local_crops_size=60,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # self.trans=transforms.Compose([
        #     LogTransform(),
        #     transforms.Normalize(mean=[3.1216, 3.1217, 3.1217, 3.1218, 3.1220],
        #                          std=[0.0064, 0.0075, 0.0096, 0.0111, 0.0170])
        # ])

        # 高斯模糊的std
        self.std_ranges = []
        self.fwhm_range = []

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomCrop(global_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomCrop(local_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        global_transfo1_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=1.0),
                RandomGaussianNoise(p=1.0, im_dim=global_crops_size),
            ]
        )

        global_transfo2_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.1),
                RandomGaussianNoise(p=0.1, im_dim=global_crops_size),
            ]
        )

        local_transfo_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.5),
                RandomGaussianNoise(p=0.5, im_dim=local_crops_size),
            ]
        )

        trans =transforms.Normalize(mean=[0.003918, 0.00564221, 0.00870101, 0.01226911, 0.01992336],
                             std=[0.4310049, 0.27417892, 0.38182727, 0.47646362, 1.3710657])

        self.global_transfo1 = transforms.Compose([global_transfo1_extra, trans])
        self.global_transfo2 = transforms.Compose([global_transfo2_extra, trans])
        self.local_transfo = transforms.Compose([local_transfo_extra, trans])

        # --- 2. Setup Photometry Augmentation and Preprocessing Pipeline ---
        # 实例化只做增强和特征工程的变换器
        self.photometry_transform = PhotometryTransform(split="train",
                                                        mean=[],
                                                        std=[])

        def __call__(self, data):
            output = {}
            image = data['image']
            params = data['params']
            # global crops:
            im1_base = self.geometric_augmentation_global(image)
            global_crop_1 = self.global_transfo1(im1_base)

            im2_base = self.geometric_augmentation_global(image)
            global_crop_2 = self.global_transfo2(im2_base)

            output["global_crops"] = [global_crop_1, global_crop_2]

            # global crops for teacher:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

            # local crops:
            local_crops = [

                self.local_transfo(self.geometric_augmentation_local(image))

                for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

            # --- 处理测光数据 ---
            photometry_global_view1 = self.photometry_transform(params)
            photometry_global_view2 = self.photometry_transform(params)

            # 2. 为所有局部视图生成独立的测光增强版本
            photometry_local_views = [
                self.photometry_transform(params)
                for _ in range(self.local_crops_number)
            ]

            output["photometry_global_views"] = [photometry_global_view1, photometry_global_view2]
            output["photometry_local_views"] = photometry_local_views
            return output

class RandomGaussianBlur(transforms.RandomApply):
    """Randomly apply Gaussian blur to the image."""

    def __init__(self, *, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianBlur()
        super().__init__([transform], p=keep_p)


class RandomGaussianNoise(transforms.RandomApply):
    """Randomly apply Gaussian noise to the image."""

    def __init__(self, *, im_dim=64, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianNoise(im_dim=im_dim)
        super().__init__([transform], p=keep_p)


class GaussianNoise:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        mean: float = 0,
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array(
            [0.1745501011610031, 0.13902300596237183, 0.11488369852304459, 0.13054630160331726, 0.17551860213279724])
        self.loc_dist = np.array([0, 0, 0, 0, 0])
        self.scale_dist = np.array(
            [0.04473619908094406, 0.016824299469590187, 0.026677900925278664, 0.04432990029454231, 0.17248259484767914])

        self.sigma_dist = np.log(self.scale_dist)

        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.031167, 0.0132974, 0.0216753, 0.0336255, 0.1219355])
        self.noise_ch_max = np.array([0.0696115, 0.024097, 0.0360224, 0.0607057, 0.2605831])

    def __call__(self, image: np.ndarray):
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] += np.random.normal(
                    self.mean, self.sigma_augment[i], size=(self.im_dim, self.im_dim)
                )

        return image


class GaussianBlur:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.3490406, 0.2582798, 0.3337356, 0.4394641, 0.6281115])
        self.loc_dist = np.array([0.752382, 0.7642949, 0.803989, 0.9436479, 1.1158381])
        self.scale_dist = np.array([1.4370568, 1.9462386, 1.6218294, 1.2486718, 0.8676406])

        self.sigma_dist = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3959353, 1.8118689, 1.5144078, 1.3738828, 1.3474936])
        self.psf_ch_max = np.array([4.1464071, 4.6310138, 4.7327875, 4.7048585, 4.4890077])

    def __call__(self, image: np.ndarray):
        # noise in channels is uncorrelated, as images taken at different times/telescopes
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] = skimage.filters.gaussian(
                    image[i, :, :], sigma=self.sigma_augment[i], mode="reflect"
                )

        return image