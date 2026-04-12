# analyze_parallel.py (最终并行优化版)
import argparse
import os
import numpy as np
from scipy.stats import lognorm
from tqdm import tqdm
from datasets import load_from_disk
import multiprocessing as mp  # <-- 导入多进程库

from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog
import warnings  # <-- 在函数顶部或脚本全局顶部导入
from photutils.utils import NoDetectionsWarning  # <-- 导入特定的警告类型

from astroclip import format_with_env

# ==============================================================================
# --- 用户配置区 ---
# ==============================================================================
IMAGE_COLUMN_NAME = "image"
NUM_CHANNELS = 5
DETECTION_THRESHOLD_SIGMA = 3.0
N_PIXELS_CONNECTED = 5

ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")


# ==============================================================================
# --- 工作函数 (由每个子进程独立执行) ---
# ==============================================================================

def process_chunk(dataset_chunk):
    """
    这个函数是并行化处理的核心。每个工作进程都会执行这个函数。
    它接收一小部分数据集，并返回对这部分数据的分析结果。
    """
    # 为这个工作进程初始化结果存储列表
    local_fwhm_unfiltered = []
    local_fwhm_measurements = [[] for _ in range(NUM_CHANNELS)]
    local_noise_measurements = [[] for _ in range(NUM_CHANNELS)]

    for example in dataset_chunk:
        try:
            image_tensor = example[IMAGE_COLUMN_NAME]
            data = image_tensor.numpy()

            if data.shape[0] != NUM_CHANNELS:
                continue

            for c in range(NUM_CHANNELS):
                channel_data = data[c, :, :]

                try:
                    bkg_estimator = MedianBackground()
                    bkg = Background2D(channel_data, (16, 16), filter_size=(3, 3), bkg_estimator=bkg_estimator)
                    noise_std = bkg.background_rms
                    local_noise_measurements[c].append(noise_std)
                    data_subtracted = channel_data - bkg.background
                except Exception:
                    continue

                detect_threshold = DETECTION_THRESHOLD_SIGMA * noise_std

                # --- 在这里添加警告过滤器 ---
                with warnings.catch_warnings():
                    # 我们告诉Python，在这个 'with' 代码块里，请忽略 NoDetectionsWarning
                    warnings.simplefilter('ignore', NoDetectionsWarning)

                    # 现在，把可能会产生警告的代码放在这个块里
                    segment_map = detect_sources(data_subtracted, detect_threshold, npixels=N_PIXELS_CONNECTED)

                # segment_map = detect_sources(data_subtracted, detect_threshold, npixels=N_PIXELS_CONNECTED)

                if segment_map is None:
                    continue

                try:
                    cat = SourceCatalog(data_subtracted, segment_map)
                    valid_point_sources = []
                    for source in cat:
                        sigma = (source.semimajor_sigma + source.semiminor_sigma) / 2.0
                        fwhm = (sigma * 2.35482).value

                        local_fwhm_unfiltered.append(fwhm)

                        if 0.5 < fwhm < 20.0:
                            brightness = source.max_value
                            valid_point_sources.append((brightness, fwhm))

                    if not valid_point_sources:
                        continue

                    valid_point_sources.sort(key=lambda x: x[0], reverse=True)
                    brightest_fwhm = valid_point_sources[0][1]
                    local_fwhm_measurements[c].append(brightest_fwhm)
                except Exception:
                    continue
        except Exception:
            continue

    # 返回这个进程处理的所有结果
    return local_fwhm_unfiltered, local_fwhm_measurements, local_noise_measurements


# ==============================================================================
# --- 主逻辑 (由主进程执行) ---
# ==============================================================================

def analyze_data_parallel(dset_dir: str, num_workers: int):
    dataset_dir = os.path.join(ASTROCLIP_ROOT, "data", dset_dir, "train_dataset")

    if not os.path.isdir(dataset_dir):
        print(f"错误: Hugging Face 数据集目录 '{dataset_dir}' 未找到。")
        return
    print(f"正在处理数据集 '{dset_dir}'...")

    print(f"检测到 {mp.cpu_count()} 个CPU核心。将使用 {num_workers} 个工作进程进行并行分析。")

    # --- 1. 数据加载与分块 ---
    print("--- 步骤 1: 正在加载数据集并进行分块... ---")
    train_dataset = load_from_disk(dataset_dir)
    train_dataset.set_format("torch", columns=[IMAGE_COLUMN_NAME])
    total_size = len(train_dataset)

    # 将数据集索引切分成 NUM_WORKERS 块
    chunk_size = total_size // num_workers
    indices = list(range(total_size))
    chunks_of_indices = [indices[i:i + chunk_size] for i in range(0, total_size, chunk_size)]

    # 根据索引块创建数据集子集
    dataset_chunks = [train_dataset.select(chunk_indices) for chunk_indices in chunks_of_indices]

    print(f"数据集已加载并切分为 {len(dataset_chunks)} 块，每块约 {chunk_size} 个样本。")

    # --- 2. 并行处理 ---
    print("\n--- 步骤 2: 开始并行处理，这可能需要一些时间... ---")

    # 创建一个进程池
    pool = mp.Pool(processes=num_workers)

    # 使用 starmap_async 来分配任务并显示进度条
    # imap_unordered 通常效率最高，因为它不关心结果的返回顺序
    results = []
    with tqdm(total=len(dataset_chunks), desc="并行处理进度") as pbar:
        for result in pool.imap_unordered(process_chunk, dataset_chunks):
            results.append(result)
            pbar.update()

    # 关闭进程池，等待所有进程结束
    pool.close()
    pool.join()

    print("\n--- 步骤 2 完成: 所有工作进程已结束 ---")

    # --- 3. 结果汇总 ---
    print("\n--- 步骤 3: 正在汇总所有进程的分析结果... ---")

    all_fwhm_unfiltered = []
    psf_fwhm_measurements = [[] for _ in range(NUM_CHANNELS)]
    noise_std_measurements = [[] for _ in range(NUM_CHANNELS)]

    for res_unfiltered, res_fwhm, res_noise in results:
        all_fwhm_unfiltered.extend(res_unfiltered)
        for c in range(NUM_CHANNELS):
            psf_fwhm_measurements[c].extend(res_fwhm[c])
            noise_std_measurements[c].extend(res_noise[c])

    print("结果汇总完成。")

    # --- 4. 最终分析与拟合 (这部分和之前完全一样) ---
    print("\n--- 步骤 4: 开始最终的全局分析与参数拟合 ---")

    print("--- 步骤 4.1: 开始全局FWHM分布分析，确定最佳过滤范围 ---")

    # 将所有通道的FWHM值合并到一个大列表中用于全局分析
    all_fwhm_global = np.array(all_fwhm_unfiltered)
    all_fwhm_global_clean = all_fwhm_global[~np.isnan(all_fwhm_global)]
    # recommended_min, recommended_max = 0.5, 20.0  # 设定默认值

    if len(all_fwhm_global_clean) > 0:
        num_bins = 200
        hist_range = (0, 30)
        counts, bin_edges = np.histogram(all_fwhm_global_clean, bins=num_bins, range=hist_range)
        peak_index = np.argmax(counts)
        peak_fwhm_value = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2.0

        median_fwhm_value = np.median(all_fwhm_global_clean)

        # 根据分析结果，确定最终的过滤范围
        recommended_min = max(0.5, peak_fwhm_value * 0.5)
        recommended_max = peak_fwhm_value * 2.5  # 使用2.5倍作为更稳健的上限

        print(f"全局分析找到的主峰位于 FWHM = {peak_fwhm_value:.2f} 像素")
        print(f"全局FWHM值的中位数为: {median_fwhm_value:.2f} 像素")
        print(f"\n已确定全局最佳过滤范围: ({recommended_min:.2f}, {recommended_max:.2f})")

        # 可视化
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 7))
            plt.hist(all_fwhm_global_clean, bins=num_bins, range=hist_range, label='全局FWHM分布')
            plt.axvline(peak_fwhm_value, color='r', linestyle='--', linewidth=2,
                        label=f'peak value ≈ {peak_fwhm_value:.2f} pix')
            plt.axvline(median_fwhm_value, color='g', linestyle=':', linewidth=2,
                        label=f'median value ≈ {median_fwhm_value:.2f} pix')
            plt.axvspan(recommended_min, recommended_max, color='orange', alpha=0.2, label='recommended filtered range')
            plt.xlabel("FWHM (pixels)");
            plt.ylabel("源的数量 (Number of sources)");
            plt.title("train_dataset FWHM分布直方图")
            plt.legend();
            plt.grid(True, alpha=0.5);
            plt.yscale('log')
            plot_filename = "../fwhm_distribution_final.png"
            plt.savefig(plot_filename)
            print(f"直方图已保存为 '{plot_filename}'，请查看。")
        except ImportError:
            print("提示: 安装 matplotlib (`pip install matplotlib`) 以查看可视化直方图。")

    else:
        print("未能收集到任何FWHM值，将使用默认过滤范围 (0.5, 20.0)。")

    # --- 第3步: 使用最佳范围进行逐通道筛选与拟合 ---
    print("--- 步骤 4.2: 开始使用最佳范围进行逐通道筛选与最终参数拟合 ---")

    # --- 用于 GaussianBlur (PSF) 的最终拟合 ---
    print("\n" + "=" * 50)
    print("--- 用于 GaussianBlur 类 (PSF 大小) 的参数 ---")
    print("=" * 50)

    psf_shapes, psf_locs, psf_scales = [], [], []
    psf_mins, psf_maxs = [], []
    for c in range(NUM_CHANNELS):
        # 从该通道的 *未经过滤* 数据开始
        data_to_filter = np.array(psf_fwhm_measurements[c])
        data_to_fit_clean = data_to_filter[~np.isnan(data_to_filter)]

        # 使用全局最佳范围进行筛选
        data_to_fit = data_to_filter[(data_to_fit_clean > recommended_min) & (data_to_fit_clean < recommended_max)]

        print(f"通道 {c}: 原始FWHM点数 {len(data_to_fit_clean)}, 筛选后点数 {len(data_to_fit)}")

        if len(data_to_fit) < 20:
            print(f"  -> PSF数据点不足，无法拟合。将使用默认值。")
            psf_shapes.append(1.0);
            psf_locs.append(1.0);
            psf_scales.append(1.0)
            psf_mins.append(1.0);
            psf_maxs.append(5.0)
            continue

        shape, loc, scale = lognorm.fit(data_to_fit)
        psf_shapes.append(shape);
        psf_locs.append(loc);
        psf_scales.append(scale)
        psf_mins.append(np.percentile(data_to_fit, 1))
        psf_maxs.append(np.percentile(data_to_fit, 99))

    print("\n请将以下数组复制到你的 GaussianBlur 类的 __init__ 方法中:")
    print(f"self.shape_dist = np.array({np.round(psf_shapes, 7).tolist()})")
    print(f"self.loc_dist = np.array({np.round(psf_locs, 7).tolist()})")
    print(f"self.scale_dist = np.array({np.round(psf_scales, 7).tolist()})")
    print("\n# (可选) 你可以使用以下值为 psf_ch_min 和 psf_ch_max 赋值")
    print(f"self.psf_ch_min = np.array({np.round(psf_mins, 7).tolist()})")
    print(f"self.psf_ch_max = np.array({np.round(psf_maxs, 7).tolist()})")

    # --- 用于 GaussianNoise (背景噪声) 的最终拟合 ---
    print("\n" + "=" * 50)
    print("--- 用于 GaussianNoise 类 (背景噪声) 的参数 ---")
    print("=" * 50)

    noise_shapes, noise_locs, noise_scales = [], [], []
    noise_mins, noise_maxs = [], []
    # --- 新增: 创建一个列表来存储每个通道的中位数 ---
    noise_medians = []

    for c in range(NUM_CHANNELS):
        data_to_fit = np.array(noise_std_measurements[c])
        # 移除 <= 0 的值 (防止对数计算错误，虽然噪声一般为正)
        data_to_fit = data_to_fit[data_to_fit > 1e-9]

        print(f"通道 {c}: 噪声点数 {len(data_to_fit)}")
        if len(data_to_fit) < 20:
            print(f"  -> 噪声数据点不足，无法拟合。将使用默认值。")
            noise_shapes.append(0.1);
            noise_locs.append(0.0);
            noise_scales.append(0.01)
            noise_mins.append(0.001);
            noise_maxs.append(0.1)
            continue

        # --- 优化策略 1: 降采样 (极大地加速拟合) ---
        # 如果数据点超过 10,000 个，只随机取 10,000 个来拟合参数，这足够代表分布了
        if len(data_to_fit) > 10000:
            fit_subset = np.random.choice(data_to_fit, size=10000, replace=False)
        else:
            fit_subset = data_to_fit

        shape, loc, scale = lognorm.fit(fit_subset)
        # shape, loc, scale = lognorm.fit(data_to_fit)
        # # --- 核心修改: 优先尝试三参数拟合，失败则回退 ---
        # try:
        #     # 尝试使用更稳健的 'Nelder-Mead' 优化器进行三参数拟合
        #     shape, loc, scale = lognorm.fit(data_to_fit, optimizer='Nelder-Mead')
        # except (ValueError, RuntimeError, FloatingPointError):
        #     # 如果仍然失败，则回退到固定loc=0的稳定拟合
        #     print(f"  -> 通道 {c}: 三参数拟合失败，回退到固定loc=0的稳定拟合。")
        #     shape, loc, scale = lognorm.fit(data_to_fit, floc=0)

        noise_shapes.append(shape);
        noise_locs.append(loc);
        noise_scales.append(scale)
        noise_mins.append(np.percentile(data_to_fit, 1))
        noise_maxs.append(np.percentile(data_to_fit, 99))
        # --- 新增: 计算并存储当前通道的中位数 ---
        noise_medians.append(np.median(data_to_fit))

    print("\n请将以下数组复制到你的 GaussianNoise 类的 __init__ 方法中:")
    print(f"self.shape_dist = np.array({np.round(noise_shapes, 7).tolist()})")
    print(f"self.loc_dist = np.array({np.round(noise_locs, 7).tolist()})")
    print(f"self.scale_dist = np.array({np.round(noise_scales, 7).tolist()})")
    print("\n# (可选) 你可以使用以下值为 noise_ch_min 和 noise_ch_max 赋值")
    print(f"self.noise_ch_min = np.array({np.round(noise_mins, 7).tolist()})")
    print(f"self.noise_ch_max = np.array({np.round(noise_maxs, 7).tolist()})")
    print("\n" + "=" * 50)

    # --- 新增: 打印用于AsinhStretch的softening_factor ---
    print("\n" + "=" * 50)
    print("--- 用于 AsinhStretch 类的 softening_factor 参数 ---")
    print("=" * 50)
    print("推荐使用每个通道背景噪声的中位数作为 softening_factor。")
    print("\n请在实例化 AsinhStretch 时使用以下列表:")
    print(f"softening_factor = {np.round(noise_medians, 7).tolist()}")
    print("\n" + "=" * 50)

    # ==============================================================================
    # --- 新增: 验证拟合结果的可视化代码 ---
    # ==============================================================================
    print("\n" + "=" * 50)
    print("--- 正在生成拟合验证图 (validation_noise_fit.png) ---")
    print("=" * 50)

    try:

        # 创建一个大图，包含5个子图 (对应5个通道)
        fig, axes = plt.subplots(1, NUM_CHANNELS, figsize=(20, 4))
        if NUM_CHANNELS == 1: axes = [axes]  # 兼容单通道情况

        for c in range(NUM_CHANNELS):
            ax = axes[c]

            # 1. 获取原始数据 (为了画图清爽，过滤掉极端异常值)
            data = np.array(noise_std_measurements[c])
            data = data[data > 1e-9]  # 去除0
            p99 = np.percentile(data, 99.5)
            data_clean = data[data < p99]  # 只画99.5%的数据，避免长尾把图拉得太长看不清

            # 2. 画原始数据的直方图 (Density=True 表示归一化面积为1)
            ax.hist(data_clean, bins=100, density=True, alpha=0.6, color='skyblue', label='Raw Data')

            # 3. 画拟合的曲线
            # 获取该通道拟合出的参数
            shape = noise_shapes[c]
            loc = noise_locs[c]  # 这里应该是 0
            scale = noise_scales[c]

            # 生成平滑的 x 轴坐标
            x = np.linspace(data_clean.min(), data_clean.max(), 1000)
            # 计算对应的概率密度 y
            pdf = lognorm.pdf(x, shape, loc, scale)

            # 画线
            ax.plot(x, pdf, 'r-', lw=2, label=f'Fit: s={shape:.2f}\nloc={loc:.2f}, scale={scale:.3f}')

            ax.set_title(f'Channel {c}')
            ax.legend(fontsize='small')
            ax.set_xlabel('Noise Sigma')

        plt.tight_layout()
        plt.savefig("validation_noise_fit.png", dpi=150)
        print(f"验证图片已保存为: {os.getcwd()}/validation_noise_fit.png")
        print("请打开图片检查：红线是否紧贴蓝色直方图？")

    except Exception as e:
        print(f"绘图过程中出错: {e}")

    # ==============================================================================
    # --- 新增: 验证 PSF (FWHM) 拟合结果的可视化代码 ---
    # ==============================================================================
    print("\n" + "=" * 50)
    print("--- 正在生成 PSF 拟合验证图 (validation_psf_fit.png) ---")
    print("=" * 50)

    try:
        # 创建一个大图，包含5个子图 (对应5个通道)
        fig, axes = plt.subplots(1, NUM_CHANNELS, figsize=(20, 4))
        if NUM_CHANNELS == 1: axes = [axes]

        for c in range(NUM_CHANNELS):
            ax = axes[c]

            # 1. 获取原始数据
            # 注意：我们需要重现拟合时的数据筛选逻辑，否则图和曲线对不上
            # 拟合时我们用了 recommended_min 和 recommended_max
            data = np.array(psf_fwhm_measurements[c])
            data = data[~np.isnan(data)]  # 去除 NaN

            # 为了画图清晰，我们只画出拟合范围附近的数据
            # (之前的逻辑是 filtered data，这里我们画出核心区域 99% 的数据)
            p01 = np.percentile(data, 1)
            p99 = np.percentile(data, 99)

            # 筛选画图用的数据 (去除极端的离群值，让直方图更好看)
            data_plot = data[(data > p01) & (data < p99)]

            # 2. 画数据的直方图
            # bins=50 足够看清形状
            ax.hist(data_plot, bins=50, density=True, alpha=0.6, color='lightgreen', label='Observed FWHM')

            # 3. 画拟合的曲线
            shape = psf_shapes[c]
            loc = psf_locs[c]
            scale = psf_scales[c]

            # 生成 x 轴坐标 (覆盖数据的范围)
            x = np.linspace(data_plot.min(), data_plot.max(), 1000)

            # 计算 PDF
            pdf = lognorm.pdf(x, shape, loc, scale)

            # 画线
            ax.plot(x, pdf, 'b-', lw=2, label=f'Fit: s={shape:.2f}\nloc={loc:.2f}, sc={scale:.2f}')

            ax.set_title(f'Channel {c} PSF')
            ax.legend(fontsize='small')
            ax.set_xlabel('FWHM (pixels)')

            # 加上辅助线，标示之前的推荐过滤范围
            ax.axvline(recommended_min, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(recommended_max, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig("validation_psf_fit.png", dpi=150)
        print(f"验证图片已保存为: {os.getcwd()}/validation_psf_fit.png")
        print("请检查：蓝色曲线是否贴合绿色直方图？(特别注意峰值位置)")

    except Exception as e:
        print(f"PSF 绘图过程中出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Analysis of Astronomical Images for PSF and Noise Statistics")

    # 路径相关
    parser.add_argument(
        "--dset_dir",
        type=str,
        default="data_g3_z",
        help="完整的数据集路径 (包含 train_dataset 文件夹)"
    )

    # 并行配置
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes"
    )

    args = parser.parse_args()

    # 启动分析
    analyze_data_parallel(args.dset_dir, args.num_workers)
