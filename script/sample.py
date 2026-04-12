import numpy as np
from datasets import load_from_disk, DatasetDict
import os
import shutil


def create_sample_dataset(train_dir, test_dir, output_dir, sample_size=1000, random_seed=42):
    """
    从训练集和测试集中随机选取指定数量的数据，创建新的数据集

    Parameters:
    - train_dir: 训练集路径
    - test_dir: 测试集路径
    - output_dir: 输出路径
    - sample_size: 每个数据集选取的样本数量
    - random_seed: 随机种子
    """

    # 设置随机种子确保可重复性
    np.random.seed(random_seed)

    # 加载训练集和测试集
    print("正在加载训练集...")
    train_dataset = load_from_disk(train_dir)
    print(f"训练集大小: {len(train_dataset)}")

    print("正在加载测试集...")
    test_dataset = load_from_disk(test_dir)
    print(f"测试集大小: {len(test_dataset)}")

    # 检查数据集大小是否足够
    if len(train_dataset) < sample_size:
        print(f"警告: 训练集大小({len(train_dataset)})小于请求的样本数({sample_size})")
        train_sample_size = len(train_dataset)
    else:
        train_sample_size = sample_size

    if len(test_dataset) < sample_size:
        print(f"警告: 测试集大小({len(test_dataset)})小于请求的样本数({sample_size})")
        test_sample_size = len(test_dataset)
    else:
        test_sample_size = sample_size

    # 随机选取训练集样本
    print(f"正在从训练集中随机选取{train_sample_size}个样本...")
    train_indices = np.random.choice(len(train_dataset), train_sample_size, replace=False)
    train_sample = train_dataset.select(train_indices)

    # 随机选取测试集样本
    print(f"正在从测试集中随机选取{test_sample_size}个样本...")
    test_indices = np.random.choice(len(test_dataset), test_sample_size, replace=False)
    test_sample = test_dataset.select(test_indices)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存采样的数据集
    print("正在保存采样的训练集...")
    train_output_path = os.path.join(output_dir, "train_dataset")
    train_sample.save_to_disk(train_output_path)

    print("正在保存采样的测试集...")
    test_output_path = os.path.join(output_dir, "test_dataset")
    test_sample.save_to_disk(test_output_path)


    print(f"数据集已保存到: {output_dir}")
    print(f"采样训练集大小: {len(train_sample)}")
    print(f"采样测试集大小: {len(test_sample)}")

    # 输出数据集信息
    # print("\n=== 数据集信息 ===")
    # print(f"训练集列名: {train_sample.column_names}")
    # if 'image' in train_sample.column_names:
    #     sample_image = train_sample[0]['image']
    #     print(f"Image shape: {sample_image.shape}")
    # if 'spectrum' in train_sample.column_names:
    #     sample_spectrum = train_sample[0]['spectrum']
    #     print(f"Spectrum shape: {sample_spectrum.shape}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    root = "/home/kongxiao/data45/kx/dsm"  # 根据你的实际路径修改
    dset_dir = f"{root}/data/data_67W"
    train_dir = f"{dset_dir}/train_dataset"
    test_dir = f"{dset_dir}/test_dataset"  # 假设测试集路径
    output_dir = f"{dset_dir}/sample_dataset"

    # 创建采样数据集
    create_sample_dataset(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        sample_size=1000,
        random_seed=42
    )
