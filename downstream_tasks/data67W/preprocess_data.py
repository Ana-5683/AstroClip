import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from datasets import load_from_disk, Dataset, Sequence, Value
import logging

from astroclip import format_with_env

# --- 设置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.mean = torch.tensor([20.9822, 20.5111, 20.3487, 20.2235, 20.0923, -2.1290, -2.3785, -2.7470,
                                  -3.0439, -3.3398, -2.3635, -3.3634, -3.3151, -3.1305, -2.1478, 0.4691,
                                  0.1642, 0.1257, 0.1308])
        self.std = torch.tensor([1.2187, 0.9515, 0.9641, 0.9652, 0.9636, 0.6783, 0.6783, 0.6783, 0.6783,
                                 0.6783, 0.8244, 0.4466, 0.5557, 0.6140, 0.7735, 0.6593, 0.2383, 0.1920,
                                 0.2736])

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


# --- 核心预处理函数 ---
def create_preprocessing_function(size):
    """
    创建一个闭包函数，用于处理数据集中的每个样本。
    这样做可以预先构建好图像变换流程，提高效率。
    """
    # 1. 定义图像处理流程
    # 完全按照您提供的参数和流程
    image_transform = transforms.Compose([
        transforms.CenterCrop(size),
        # z0-6
        # transforms.Normalize(mean=[0.00423534, 0.00681197, 0.01025112, 0.01426154, 0.02135192],
        #                      std=[0.22170926, 0.28257167, 0.3946128, 0.48931952, 1.21125847])

        # z1-5
        transforms.Normalize(mean=[0.00392827, 0.00634693, 0.00951248, 0.01321839, 0.01975857],
                             std=[0.20640903, 0.27777484, 0.38276206, 0.4807447, 1.12889713])
    ])

    photo_transform = PhotometryTransform()

    def preprocess(example: dict) -> dict:
        """
        处理单个数据样本 (一个字典)。
        """

        # 1. 处理测光
        example['params'] = photo_transform(example['params']).numpy()
        # example['params']=example['params'].numpy()
        # 2. 处理图像
        # 将处理后的 Tensor 转回 NumPy 数组以便 datasets 库保存
        example['image'] = image_transform(example['image']).numpy()

        # 3. 处理光谱
        example['spectrum'] = example['spectrum'].numpy()

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
        input_path = os.path.join(args.dataset_dir, folder_name)
        output_path = f"{input_path}_preprocessed_{args.name}"

        if not os.path.exists(input_path):
            logging.warning(f"跳过: 未找到 {split} 数据集路径 {input_path}")
            continue

        logging.info(f"--- 正在处理 {split} 数据集 ---")
        logging.info(f"加载数据集于: {input_path}")

        # 1. 加载数据集
        original_dataset = load_from_disk(input_path)
        logging.info("原始数据集信息:")
        print(original_dataset)
        original_dataset.set_format("torch")

        # 2. 应用预处理
        logging.info("开始进行预处理... (这可能需要一些时间)")
        # 使用 .map() 方法高效地应用处理函数，num_proc 利用多核加速
        processed_dataset = original_dataset.map(
            processing_func,
            num_proc=args.num_workers
        )

        # --- 【修改方案】使用通用的 Sequence 定义，不再限制具体形状 ---
        logging.info("正在将元数据修改为通用格式 (Sequence)...")

        new_features = processed_dataset.features.copy()

        # 将 image 定义为：任意大小的 3D float32 列表
        # 结构: [Dim1, Dim2, Dim3] -> Sequence(Sequence(Sequence(Value)))
        new_features["image"] = Sequence(Sequence(Sequence(Value("float32"))))

        # 将 params 定义为：任意长度的 1D float32 列表 (因为从15变成了19)
        new_features["params"] = Sequence(Value("float32"))

        # 应用修改 (Cast)
        processed_dataset = processed_dataset.cast(new_features)
        # --------------------------------------------------------

        logging.info("预处理完成。")
        logging.info("处理后数据集信息:")
        print(processed_dataset)

        # 3. 保存处理后的数据集
        logging.info(f"保存处理后的数据集到: {output_path}")
        processed_dataset.save_to_disk(output_path, num_proc=args.num_workers)
        logging.info(f"{split} 数据集已成功处理并保存。")
        print("\n")


if __name__ == "__main__":
    # 从环境变量获取 ASTROCLIP_ROOT 的默认值
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")

    parser = argparse.ArgumentParser(
        description="对 AstroCLIP 的图像和光谱数据进行标准化预处理。"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=f"{ASTROCLIP_ROOT}/data/data_67W",
        help="AstroCLIP 项目的根目录。默认会从环境变量 ASTROCLIP_ROOT 读取，如果未设置则为当前目录。"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        # default="c48",
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
