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

        #67W
        # self.mean = torch.tensor([20.9822, 20.5111, 20.3487, 20.2235, 20.0923, -2.1290, -2.3785, -2.7470,
        #                           -3.0439, -3.3398, -2.3635, -3.3634, -3.3151, -3.1305, -2.1478, 0.4691,
        #                           0.1642, 0.1257, 0.1308])
        # self.std = torch.tensor([1.2187, 0.9515, 0.9641, 0.9652, 0.9636, 0.6783, 0.6783, 0.6783, 0.6783,
        #                          0.6783, 0.8244, 0.4466, 0.5557, 0.6140, 0.7735, 0.6593, 0.2383, 0.1920,
        #                          0.2736])

        self.log_indices = log_transform_indices
        self.epsilon = epsilon

        # --- Define indices for clarity based on the 15-dim input tensor ---
        self.mag_indices = slice(0, 5)
        self.ext_indices = slice(5, 10)
        self.err_indices = slice(10, 15)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Applies the full transformation pipeline to a single 15-dim tensor.

        Args:
            sample (torch.Tensor): A tensor of shape (15,) with features in the specified order.

        Returns:
            torch.Tensor: A 19-dimensional, preprocessed tensor ready for the model.
        """
        # --- Validate input tensor shape ---
        if sample.shape != (15,):
            raise ValueError(f"Input tensor must have shape (15,), but got {sample.shape}")

        # --- 1. Feature Engineering ---

        # 1a. Extinction Correction
        corrected_mags = sample[self.mag_indices] - sample[self.ext_indices]

        # 1b. Create Color Features
        # u-g, g-r, r-i, i-z
        colors = corrected_mags[:-1] - corrected_mags[1:]

        # --- Assemble the final 19-dimensional vector ---
        # The order defined here is CRITICAL and must be the same one used
        # to calculate the mean and std vectors.
        # Order: 5 corrected mags, 5 extinctions, 5 errors, 4 colors
        processed_vector = torch.cat([
            corrected_mags,
            sample[self.ext_indices],
            sample[self.err_indices],
            colors
        ])

        # --- 2. Log Transformation ---
        if self.log_indices:
            # Clone to avoid modifying the tensor in-place if it's not desired
            vec_clone = processed_vector.clone()
            vec_clone[self.log_indices] = torch.log(processed_vector[self.log_indices] + self.epsilon)
            processed_vector = vec_clone

        # --- 3. Standardization (Z-score Normalization) ---
        # Add epsilon to std to prevent division by zero in case std is 0 for some feature
        standardized_vector = (processed_vector - self.mean) / (self.std + self.epsilon)

        return standardized_vector
