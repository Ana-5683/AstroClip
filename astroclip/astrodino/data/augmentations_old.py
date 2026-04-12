# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import List, Tuple

import numpy as np
import skimage.filters
import skimage.transform
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

logger = logging.getLogger("dinov2")


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

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomRotation(180),
                transforms.RandomCrop(global_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomRotation(180),
                transforms.RandomCrop(local_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        global_transfo1_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=1.0),
                RandomGaussianNoise(p=1.0, im_dim=global_crops_size),
                # RandomFluxJitter(p=1.0)
            ]
        )

        global_transfo2_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.1),
                RandomGaussianNoise(p=0.1, im_dim=global_crops_size),
                # RandomFluxJitter(p=0.1)
            ]
        )

        local_transfo_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.5),
                RandomGaussianNoise(p=0.5, im_dim=local_crops_size),
                # RandomFluxJitter(p=0.5)
            ]
        )

        self.global_transfo1 = transforms.Compose([global_transfo1_extra])
        self.global_transfo2 = transforms.Compose([global_transfo2_extra])
        self.local_transfo = transforms.Compose([local_transfo_extra])

        # now
        self.norm = transforms.Normalize(mean=[0.003918, 0.00564221, 0.00870101, 0.01226911, 0.01992336],
                                         std=[0.4310049, 0.27417892, 0.38182727, 0.47646362, 1.3710657])

        # data67W

        # z1-5
        # self.norm = transforms.Normalize(mean=[0.00392827, 0.00634693, 0.00951248, 0.01321839, 0.01975857],
        #                                  std=[0.20640903, 0.27777484, 0.38276206, 0.4807447, 1.12889713])

    def __call__(self, image):
        output = {}

        # image=self.trans(image)

        # global crops:
        im1_base = np.array(self.geometric_augmentation_global(image))
        global_crop_1 = self.norm(torch.tensor(self.global_transfo1(im1_base)))

        im2_base = np.array(self.geometric_augmentation_global(image))
        global_crop_2 = self.norm(torch.tensor(self.global_transfo2(im2_base)))

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [

            self.norm(torch.tensor(self.local_transfo(np.array((self.geometric_augmentation_local(image))))))

            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

# class RandomFluxJitter(transforms.RandomApply):
#     """
#     Randomly apply FluxJitter to the image.
#     """
#
#     def __init__(self, p=0.5, jitter_range=(0.95, 1.05)):
#         self.keep_p = 1 - p
#         transform = FluxJitter(jitter_range)
#         super().__init__([transform], p=self.keep_p)
#
# class FluxJitter:
#     """
#     独立缩放每个波段，模拟 5% 的光度校准误差。
#     """
#
#     def __init__(self, jitter_range=(0.95, 1.05)):
#         self.min_val, self.max_val = jitter_range
#
#     def __call__(self, img):
#         # img: Tensor [C, H, W] or ndarray
#         # 假设输入是 Tensor，如果是 numpy 需要相应调整
#         C = img.shape[0]
#         scales = np.random.uniform(self.min_val, self.max_val, size=(C, 1, 1))
#         return img * scales

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
            im_dim: int = 64,
            im_ch: int = 5,
            decals: bool = True,
            uniform: bool = False,
    ):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters

        # now
        self.shape_dist = np.array(
            [0.1745501011610031, 0.13902300596237183, 0.11488369852304459, 0.13054630160331726, 0.17551860213279724])
        self.loc_dist = np.array([0, 0, 0, 0, 0])
        self.scale_dist = np.array(
            [0.04473619908094406, 0.016824299469590187, 0.026677900925278664, 0.04432990029454231, 0.17248259484767914])

        self.sigma_dist = np.log(self.scale_dist)

        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.031167, 0.0132974, 0.0216753, 0.0336255, 0.1219355])
        self.noise_ch_max = np.array([0.0696115, 0.024097, 0.0360224, 0.0607057, 0.2605831])

        #data_67W
        # self.shape_dist = np.array([0.2901305854320526, 0.4321700930595398, 0.31048551201820374, 0.1958114057779312, 0.3243331015110016])
        # self.loc_dist = np.array([0.0180391, 0.0116148, 0.0173531, 0.0156599, 0.0783548])
        # self.scale_dist = np.array([0.02613070048391819, 0.004938099998980761, 0.0090744998306036, 0.02843339927494526,
        #      0.09203749895095825])
        #
        # self.sigma_dist = np.log(self.scale_dist)
        #
        # self.noise_ch_min = np.array([0.0312619, 0.0133542, 0.021672, 0.0336232, 0.1220722])
        # self.noise_ch_max = np.array([0.069185, 0.0241808, 0.0359386, 0.0606479, 0.2610066])

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
        self.sigma_augment = self.sigma_final ** 2 - self.sigma_true ** 2
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
            im_ch: int = 5,
            decals: bool = True,
            uniform: bool = False,
    ):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters

        # now
        self.shape_dist = np.array([0.3490406, 0.2582798, 0.3337356, 0.4394641, 0.6281115])
        self.loc_dist = np.array([0.752382, 0.7642949, 0.803989, 0.9436479, 1.1158381])
        self.scale_dist = np.array([1.4370568, 1.9462386, 1.6218294, 1.2486718, 0.8676406])

        self.sigma_dist = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3959353, 1.8118689, 1.5144078, 1.3738828, 1.3474936])
        self.psf_ch_max = np.array([4.1464071, 4.6310138, 4.7327875, 4.7048585, 4.4890077])

        # data_67W
        # self.shape_dist = np.array([0.2566401, 0.2733183, 0.3251265, 0.3942789, 0.5002745])
        # self.loc_dist = np.array([0.0276982, 0.6849072, 0.6558717, 0.7608271, 0.9792627])
        # self.scale_dist = np.array([2.4549237, 2.2026109, 1.9550941, 1.6290672, 1.1643986])
        #
        # self.sigma_dist = np.log(self.scale_dist)
        #
        # self.psf_ch_min = np.array([1.4146431, 1.8498239, 1.5530287, 1.393043, 1.3624972])
        # self.psf_ch_max = np.array([4.3686745, 5.0043672, 5.0951172, 5.0614738, 4.6931877])

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
        self.sigma_augment = self.sigma_final ** 2 - self.sigma_true ** 2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                image[i, :, :] = skimage.filters.gaussian(
                    image[i, :, :], sigma=self.sigma_augment[i], mode="reflect"
                )

        return image
