import torch
from torch.utils.data import DataLoader

from astroclip.astrophoto.data import PhotometryDataset


def compute_log_statistics(data_path, batch_size=1024, num_workers=4):
    """
    计算测光数据（包括色指数）在对数变换后的均值和标准差

    Args:
        data_path (str): 数据集路径
        batch_size (int): 批处理大小
        num_workers (int): 数据加载器的工作线程数

    Returns:
        tuple: (mean, std) 均值和标准差张量
    """

    # 创建一个只做特征工程和对数变换的变换对象
    class LogStatsTransform:
        def __init__(self, epsilon=1e-6):
            self.mag_indices = slice(0, 5)      # psfMag (u,g,r,i,z)
            self.ext_indices = slice(5, 10)     # extinction (u,g,r,i,z)
            self.err_indices = slice(10, 15)    # psfMagErr (u,g,r,i,z)
            self.epsilon = epsilon

        def __call__(self, sample):
            if isinstance(sample, dict):
                sample = torch.tensor(list(sample.values()), dtype=torch.float32)

            # 验证输入张量形状
            if sample.shape != (15,):
                raise ValueError(f"输入张量必须是形状(15,)，但得到 {sample.shape}")

            # 1. 特征工程
            # 1a. 消光修正
            corrected_mags = sample[self.mag_indices] - sample[self.ext_indices]

            # 1b. 创建颜色特征
            # u-g, g-r, r-i, i-z
            colors = corrected_mags[:-1] - corrected_mags[1:]

            # 2. 对特定特征进行对数变换（extinction和psfMagErr）
            log_extinctions = torch.log(sample[self.ext_indices] + self.epsilon)
            log_errors = torch.log(sample[self.err_indices] + self.epsilon)

            # 3. 组装最终的19维向量（注意这里使用的是对数变换后的extinction和errors）
            # 顺序：5个修正星等，5个对数消光，5个对数误差，4个颜色
            processed_vector = torch.cat([
                corrected_mags,        # 5个消光修正星等
                log_extinctions,       # 5个对数消光
                log_errors,            # 5个对数误差
                colors                 # 4个颜色指数
            ])

            return processed_vector

    # 使用自定义变换创建数据集
    transform = LogStatsTransform()
    dataset = PhotometryDataset(
        data_path=data_path,
        split='train',
        transform=transform
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # 在线计算均值和标准差以节省内存
    n = 0
    mean = torch.zeros(19)
    m2 = torch.zeros(19)

    print("开始计算对数变换后的统计数据...")
    for batch in dataloader:
        # batch形状: (batch_size, 19)
        batch_size_actual = batch.shape[0]

        # 更新统计数据
        for i in range(batch_size_actual):
            n += 1
            delta = batch[i] - mean
            mean += delta / n
            delta2 = batch[i] - mean
            m2 += delta * delta2

    # 计算方差和标准差
    if n < 2:
        raise ValueError("至少需要2个样本才能计算统计信息")

    variance = m2 / (n - 1)
    std = torch.sqrt(variance)

    print(f"总共处理了 {n} 个样本")
    print("特征顺序:")
    print("- 0-4: 消光修正星等 (corrected magnitudes)")
    print("- 5-9: 对数消光 (log extinctions)")
    print("- 10-14: 对数误差 (log errors)")
    print("- 15-18: 颜色指数 (colors)")
    print(f"均值: {mean}")
    print(f"标准差: {std}")

    return mean, std

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际数据路径
    root = "/home/kongxiao/data45/kx/dsm"
    # root="/hy-tmp"
    # root = "/mnt/d/SoftWare/PycharmProjects"
    data_path = f"{root}/data/data_67W"

    try:
        mean, std = compute_log_statistics(data_path)
        print("\n计算完成！")
        print(f"Mean tensor: {mean.tolist()}")
        print(f"Std tensor: {std.tolist()}")

        # 可选：保存结果供后续使用,写入mean_std.txt中
        with open(f"./mean_std_67W.txt", "w") as f:
            f.write(f"Mean tensor: {mean.tolist()}\n")
            f.write(f"Std tensor: {std.tolist()}\n")

        print("对数变换后的统计数据已保存到文件")

    except Exception as e:
        print(f"计算过程中出错: {e}")
