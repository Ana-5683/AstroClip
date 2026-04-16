import argparse
import os
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing

from astroclip import format_with_env

# 解决多进程数据加载时可能的共享内存问题
torch.multiprocessing.set_sharing_strategy('file_system')

ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")


def calculate_image_stats(dset_dir, batch_size=64, num_workers=4):
    """
    计算图像数据集的逐通道 mean 和 std（5 个通道）
    """
    train_path = os.path.join(dset_dir, "train_dataset")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at: {train_path}")

    print(f"Loading dataset from: {train_path}")
    train_dataset = load_from_disk(train_path)
    train_dataset.set_format("torch", columns=["image"])

    print(f"Total samples: {len(train_dataset)}")

    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # 加速数据传输到GPU（即使不训练也推荐）
    )

    # 初始化累加器（使用 float64 保证精度）
    sum_per_channel = torch.zeros(5, dtype=torch.float64)
    sum_sq_per_channel = torch.zeros(5, dtype=torch.float64)
    n_pixels_per_channel = torch.zeros(5, dtype=torch.float64)

    print("Starting to compute mean and std across all images...")
    for batch in tqdm(data_loader, desc="Processing batches"):
        images = batch['image']  # shape: (B, C, H, W), C=5

        # 累加和与平方和（在 B, H, W 三个维度上求和）
        sum_per_channel += torch.sum(images, dim=[0, 2, 3]).to(torch.float64)
        sum_sq_per_channel += torch.sum(images ** 2, dim=[0, 2, 3]).to(torch.float64)

        # 每个 batch 的像素数（所有通道相同）
        n_pixels_in_batch = images.shape[0] * images.shape[2] * images.shape[3]
        n_pixels_per_channel += n_pixels_in_batch

    # 计算 mean
    mean = sum_per_channel / n_pixels_per_channel

    # 计算 std（总体标准差，与 torchvision 一致）
    var = (sum_sq_per_channel / n_pixels_per_channel) - (mean ** 2)
    var = torch.clamp(var, min=0.0)  # 防止数值误差导致负数
    std = torch.sqrt(var)

    mean_np = mean.numpy()
    std_np = std.numpy()

    print("\n" + "=" * 50)
    print("IMAGE STATISTICS CALCULATION COMPLETED!")
    print("=" * 50)
    print(f"Mean (u,g,r,i,z): {mean_np}")
    print(f"Std  (u,g,r,i,z): {std_np}")
    print("=" * 50)

    # 保存到 log.txt
    log_path = os.path.join(dset_dir, "log.txt")
    with open(log_path, "a") as f:
        f.write("\n--- Image Mean/Std Calculation ---\n")
        f.write(f"Dataset: {dset_dir}\n")
        f.write(f"Total samples: {len(train_dataset)}\n")
        f.write(f"Mean: {mean_np.tolist()}\n")
        f.write(f"Std:  {std_np.tolist()}\n")

    print(f"Results appended to: {log_path}")

    return mean_np, std_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate per-channel mean and std for Astro images")
    parser.add_argument(
        "--dset_dir",
        type=str,
        default="data_g3_z",
        help="Root directory containing 'train_dataset' folder, e.g.,data_67W_z15"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for DataLoader (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader (default: 4)"
    )

    args = parser.parse_args()

    dset_dir=os.path.join(ASTROCLIP_ROOT,"data",args.dset_dir)
    print(f"Calculating image statistics for: {dset_dir}")
    calculate_image_stats(
        dset_dir=dset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )