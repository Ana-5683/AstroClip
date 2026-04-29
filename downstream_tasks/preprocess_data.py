import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from datasets import load_from_disk, Dataset, Sequence, Value,Array3D
import logging

from astroclip import format_with_env

# --- 设置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# photo_augmentations.py

import torch
from typing import List

from torchvision.transforms import transforms

# 从环境变量获取 ASTROCLIP_ROOT 的默认值
ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")


# class PhotometryTransform:
#     """
#     A callable class that applies a series of preprocessing steps to a single
#     15-dimensional photometry tensor.
#
#     The input tensor order is assumed to be fixed:
#     - 5 psfMag (u,g,r,i,z)
#     - 5 extinction (u,g,r,i,z)
#     - 5 psfMagErr (u,g,r,i,z)
#
#     The processing steps are:
#     1. Feature Engineering:
#         a. Perform extinction correction (corrected_mag = psfMag - extinction).
#         b. Create 4 color features from corrected magnitudes.
#     2. Log Transformation: Apply log transform to specified features (errors).
#     3. Standardization: Apply Z-score normalization using pre-computed stats.
#
#     The output is a fully preprocessed 19-dimensional tensor.
#     """
#
#     def __init__(self,
#                  log_transform_indices: List[int] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
#                  epsilon: float = 1e-6):
#         """
#         Initializes the photometry transformation pipeline.
#
#         Args:
#             mean (torch.Tensor): A tensor of shape (19,) containing the pre-computed means
#                                  for the FINAL 19-dimensional feature vector.
#             std (torch.Tensor): A tensor of shape (19,) containing the pre-computed standard
#                                 deviations for the FINAL 19-dimensional feature vector.
#             log_transform_indices (List[int]): A list of indices in the FINAL 19-dim vector
#                                                that should be log-transformed. Defaults to
#                                                error features.
#             epsilon (float): A small value to add before log transform to avoid log(0).
#         """
#         self.mean = torch.tensor([21.4684, 20.9255, 20.7705, 20.6578, 20.5311, -2.1243, -2.3738, -2.7422,
#         -3.0391, -3.3351, -2.0534, -3.2229, -3.1252, -2.9056, -1.8159,  0.5431,
#          0.1548,  0.1130,  0.1263])
#         self.std = torch.tensor([0.9903, 0.5818, 0.6110, 0.6164, 0.6434, 0.6759, 0.6759, 0.6759, 0.6759,
#         0.6759, 0.7275, 0.3790, 0.4720, 0.5170, 0.5869, 0.7597, 0.2337, 0.1863,
#         0.2952])
#         self.log_indices = log_transform_indices
#         self.epsilon = epsilon
#
#         # --- Define indices for clarity based on the 15-dim input tensor ---
#         self.mag_indices = slice(0, 5)
#         self.ext_indices = slice(5, 10)
#         self.err_indices = slice(10, 15)
#
#     def __call__(self, sample: torch.Tensor) -> torch.Tensor:
#         """
#         Applies the full transformation pipeline to a single 15-dim tensor.
#
#         Args:
#             sample (torch.Tensor): A tensor of shape (15,) with features in the specified order.
#
#         Returns:
#             torch.Tensor: A 19-dimensional, preprocessed tensor ready for the model.
#         """
#         # --- Validate input tensor shape ---
#         if sample.shape != (15,):
#             raise ValueError(f"Input tensor must have shape (15,), but got {sample.shape}")
#
#         # --- 1. Feature Engineering ---
#
#         # 1a. Extinction Correction
#         corrected_mags = sample[self.mag_indices] - sample[self.ext_indices]
#
#         # 1b. Create Color Features
#         # u-g, g-r, r-i, i-z
#         colors = corrected_mags[:-1] - corrected_mags[1:]
#
#         # --- Assemble the final 19-dimensional vector ---
#         # The order defined here is CRITICAL and must be the same one used
#         # to calculate the mean and std vectors.
#         # Order: 5 corrected mags, 5 extinctions, 5 errors, 4 colors
#         processed_vector = torch.cat([
#             corrected_mags,
#             sample[self.ext_indices],
#             sample[self.err_indices],
#             colors
#         ])
#
#         # --- 2. Log Transformation ---
#         if self.log_indices:
#             # Clone to avoid modifying the tensor in-place if it's not desired
#             vec_clone = processed_vector.clone()
#             vec_clone[self.log_indices] = torch.log(processed_vector[self.log_indices] + self.epsilon)
#             processed_vector = vec_clone
#
#         # --- 3. Standardization (Z-score Normalization) ---
#         # Add epsilon to std to prevent division by zero in case std is 0 for some feature
#         standardized_vector = (processed_vector - self.mean) / (self.std + self.epsilon)
#
#         return standardized_vector


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

        # sd
        # self.mean = torch.tensor([[21.030780792236328, -2.3439688682556152], [20.533145904541016, -3.3731791973114014],
        #                           [20.37293243408203, -3.3246448040008545], [20.25986099243164, -3.1275088787078857],
        #                           [20.151731491088867, -2.1128933429718018]])
        # self.std = torch.tensor([[1.2007266283035278, 0.8237770199775696], [0.8702441453933716, 0.4253990948200226],
        #                          [0.8886646628379822, 0.52986741065979], [0.8941585421562195, 0.5942806601524353],
        #                          [0.8849909901618958, 0.7358207106590271]])

        # mine
        # self.mean = torch.tensor([[21.081636428833008, -2.303225040435791], [20.597057342529297, -3.3364884853363037],
        #                           [20.441057205200195, -3.2742390632629395], [20.323152542114258, -3.079575777053833],
        #                           [20.214853286743164, -2.0569753646850586]])
        # self.std = torch.tensor([[1.2029750347137451, 0.8231610655784607], [0.8982889652252197, 0.43984442949295044],
        #                          [0.9204807877540588, 0.5525023937225342], [0.9219300150871277, 0.6115174293518066],
        #                          [0.9159927368164062, 0.7556701898574829]])

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


def create_preprocessing_function(size):
    """
    创建一个闭包函数，用于处理数据集中的每个样本。
    这样做可以预先构建好图像变换流程，提高效率。
    """
    # 1. 定义图像处理流程
    # 完全按照您提供的参数和流程
    image_transform = transforms.Compose([
        transforms.CenterCrop(size),
        # dr16q
        transforms.Normalize(mean=[0.00383469, 0.00610499, 0.0091598, 0.01276288, 0.01901501],
                             std=[0.16391228, 0.25970103, 0.36365715, 0.45392031, 1.03993761])
        # dbx
        # transforms.Normalize(mean=[0.003918, 0.00564221, 0.00870101, 0.01226911, 0.01992336],
        #                      std=[0.4310049, 0.27417892, 0.38182727, 0.47646362, 1.3710657])

        # sd
        # transforms.Normalize(mean=[0.00379948, 0.00609522, 0.00919202, 0.01270878, 0.01900207],
        #                      std=[0.19513715, 0.26280681, 0.37203279, 0.46529404, 1.06711372])
    ])

    # 当前下游任务不使用 params，这里先保留原始字段，不做额外变换。
    # photo_transform = PhotometryTransform()

    def preprocess(example: dict) -> dict:
        """
        处理单个数据样本 (一个字典)。
        """

        # 1. 处理测光
        # example['params'] = photo_transform(example['params']).numpy().astype(np.float32, copy=False)

        # 2. 处理图像
        # 将处理后的 Tensor 转回 NumPy 数组以便 datasets 库保存
        example['image'] = image_transform(example['image']).numpy().astype(np.float32, copy=False)

        # 3. 处理光谱
        example['spectrum'] = example['spectrum'].numpy().astype(np.float32, copy=False)

        return example

    return preprocess


# --- 主执行函数 ---
def main(args):
    """
    主函数，负责加载、处理和保存数据集。
    """
    # 定义数据集类型和对应的路径
    dataset_splits = {
        'train': 'train_dataset',
        'test': 'test_dataset'
    }

    # 获取预处理函数
    processing_func = create_preprocessing_function(args.size)

    for split, folder_name in dataset_splits.items():
        # 构建输入和输出路径
        input_path = os.path.join(ASTROCLIP_ROOT, "data", args.dset, folder_name)
        output_path = f"{input_path}_preprocessed_{args.name}"

        if not os.path.exists(input_path):
            logging.warning(f"跳过: 未找到 {split} 数据集路径 {input_path}")
            continue

        logging.info(f"--- 正在处理 {split} 数据集 ---")
        logging.info(f"加载数据集于: {input_path}")

        # 1. 加载数据集
        original_dataset = load_from_disk(input_path)
        original_dataset.set_format(type="torch")
        logging.info("原始数据集信息:")
        print(original_dataset)

        print("shape信息")
        cols = ['image', 'params', 'spectrum', 'z']
        for c in cols:
            print(f"{c}:{original_dataset[c][0].shape}")

        # 2. 应用预处理
        logging.info("开始进行预处理... (这可能需要一些时间)")
        # 使用 .map() 方法高效地应用处理函数，num_proc 利用多核加速
        processed_dataset = original_dataset.map(
            processing_func,
            num_proc=args.num_workers
        )

        # 预处理后各字段的形状已经固定，直接声明成精确 schema，
        # 避免使用宽泛 Sequence 时出现 1D/2D 结构不匹配。
        logging.info("正在将元数据修改为预处理后的固定格式...")

        new_features = processed_dataset.features.copy()

        new_features["image"] = Array3D(shape=(5, args.size, args.size), dtype="float32")
        new_features["spectrum"] = Sequence(feature=Value("float32"), length=4096)

        # 应用修改 (Cast)
        processed_dataset = processed_dataset.cast(new_features)
        # --------------------------------------------------------

        logging.info("预处理完成。")
        logging.info("处理后数据集信息:")
        print(processed_dataset)

        print("shape信息")
        cols = ['image', 'params', 'spectrum', 'z']
        for c in cols:
            print(f"{c}:{processed_dataset[c][0].shape}")

        # 3. 保存处理后的数据集
        logging.info(f"保存处理后的数据集到: {output_path}")
        processed_dataset.save_to_disk(output_path, num_proc=args.num_workers)
        logging.info(f"{split} 数据集已成功处理并保存。")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="对 AstroCLIP 的图像和光谱数据进行标准化预处理。"
    )
    parser.add_argument(
        "--dset",
        type=str,
        default=f"data_sd_z15",
        help="AstroCLIP 项目的根目录。默认会从环境变量 ASTROCLIP_ROOT 读取，如果未设置则为当前目录。"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        # default="1",
        help="数据集名称。"
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        # default="48",
        help="数据集名称。"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="用于数据处理的并行进程数。"
    )

    args = parser.parse_args()
    main(args)
