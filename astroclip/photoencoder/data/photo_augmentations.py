# photo_augmentations.py
import torch
from typing import Optional


class PhotometryTransform:
    """
    Preprocessing pipeline for SDSS Photometry Data.
    Transforms raw 15-dim input into a sequence-ready (5, 2) tensor for Transformers.

    Input Shape: (15,)
    - 0-4:  psfMag (u, g, r, i, z)
    - 5-9:  extinction (u, g, r, i, z)
    - 10-14: psfMagErr (u, g, r, i, z)

    Output Shape: (5, 2)
    - Dim 0: Sequence (u, g, r, i, z)
    - Dim 1: Features [Corrected_Magnitude, Magnitude_Error]
    """

    def __init__(self,
                 epsilon: float = 1e-6):
        """
        Args:
            # mean (torch.Tensor, optional): Pre-computed mean with shape (5, 2).
            # std (torch.Tensor, optional): Pre-computed std with shape (5, 2).
            epsilon (float): Small value to avoid division by zero.
        """
        # =========================================================================
        # TODO: 必须重新计算你的数据集的 Mean 和 Std！
        # 之前的 19 维统计量不适用于现在的 (5, 2) 结构。
        # 这里的占位符意味着不做归一化 (Mean=0, Std=1)。
        # 请在训练集上计算：
        #   Mean shape should be (5, 2) -> [[u_mag_avg, u_err_avg], [g_mag_avg, g_err_avg], ...]
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

        # Indices slicing
        self.mag_indices = slice(0, 5)
        self.ext_indices = slice(5, 10)
        self.err_indices = slice(10, 15)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample (torch.Tensor): Raw 15-dim tensor.

        Returns:
            torch.Tensor: Normalized tensor of shape (5, 2).
        """
        # 1. Validation
        if sample.shape != (15,):
            raise ValueError(f"Input tensor must have shape (15,), but got {sample.shape}")

        # 2. Physics Correction: Intrinsic Mag = psfMag - extinction
        raw_mags = sample[self.mag_indices]
        extinctions = sample[self.ext_indices]
        raw_errs = sample[self.err_indices]

        corrected_mags = raw_mags - extinctions

        # --- 新增：对数变换 ---
        # 对 Error 取自然对数。加上 epsilon 防止 log(0)。
        # 注意：Magnitude 本身已经是红移的对数形式，通常不需要再 log，
        # 但 Error 是线性通量误差转化来的，建议 Log。
        log_errs = torch.log(raw_errs + self.epsilon)

        # 3. Reshape / Stacking
        # 现在的特征变成了 [Corrected_Mag, Log_MagErr]
        output_tensor = torch.stack([corrected_mags, log_errs], dim=1)

        # 4. Standardization (Z-score)
        # ！！！重要提示！！！
        # 你的 self.mean 和 self.std 必须是基于 [Mag, Log_Err] 计算出来的！
        standardized_tensor = (output_tensor - self.mean) / (self.std + self.epsilon)

        return standardized_tensor
