import bz2
import argparse
from datetime import datetime
from io import BytesIO
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import pandas as pd

from datasets import Dataset, Features, Value, Array2D, Array3D, Sequence
from tqdm import tqdm
import sys
import concurrent.futures
import multiprocessing
import warnings
from collections import defaultdict
import os

# 限制底层库线程数，防止多进程 fork 死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np

# ==========================================
# 屏蔽 Astropy 警告
# ==========================================
warnings.simplefilter('ignore', category=FITSFixedWarning)

# ==========================================
# 配置区域 (保持你的路径不变)
# ==========================================
root="/home/kongxiao/data45/kx/dsm"
CSV_FILE = f'{root}/data/DR16Q_v4.csv'
CACHE_DIR = f'{root}/cache/data_cache'
OUTPUT_DIR = f'{root}/data/dr16q_v4'

# 图像数据配置
IMAGE_ROOT = '/home/kongxiao/data44/SDSS/DR16_image/frames/301'
BANDS = ['u', 'g', 'r', 'i', 'z']
CROP_SIZE = 64
HALF_SIZE = CROP_SIZE // 2

# 光谱配置
SPEC_ROOT = '/home/kongxiao/data44/SDSS/spectra_lite/DR18'
TARGET_LEN = 4096  # 新的目标长度
WAVE_START = 3800.0  # 起始波长 (Angstrom)
WAVE_END = 9000.0  # 结束波长 (Angstrom)

# 测光数据配置
PARAMS_COLS = [
    'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z',
    'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z',
    'psfMagErr_u', 'psfMagErr_g', 'psfMagErr_r', 'psfMagErr_i', 'psfMagErr_z'
]

# 核心数设置 (保持高并发)
NUM_WORKERS = 3


# ==========================================
# 优化后的 Worker: 按 Field 处理图像
# ==========================================
def worker_process_field_group(args):
    """
    处理一个 Field 中的所有对象
    args: (run, camcol, field, objects_list, image_root_dir, save_dir)
    objects_list: list of dict [{'objID', 'ra', 'dec'}, ...]
    """
    run, camcol, field, obj_list, root_dir, save_dir = args

    # 结果统计
    processed_count = 0
    edge_count = 0
    error_ids = []
    edge_ids = []

    # ------------------------------------------------------------------
    # 【核心修改点】: 提前过滤已存在的对象
    # 在加载昂贵的 FITS 文件之前，先检查这组任务里哪些是真的需要做的
    # ------------------------------------------------------------------
    objects_to_process = []
    for obj in obj_list:
        obj_id = obj['objID']
        save_path = os.path.join(save_dir, f"{obj_id}.npy")
        if not os.path.exists(save_path):
            objects_to_process.append(obj)

    # 如果这个 Field 下的所有对象都已经存在于硬盘上，直接返回！
    # 这样就完全跳过了 BZ2 解压和 FITS 读取，速度极快
    if len(objects_to_process) == 0:
        return 0, 0, [], [], None

    # ------------------------------------------------------------------
    # 只有当确实有新任务时，才开始加载 FITS
    # ------------------------------------------------------------------

    # 1. 预先加载该 Field 的 5 个波段图像到内存
    # key: band, value: (wcs_object, image_data_numpy)
    loaded_bands = {}

    run_str = f"{run:06d}"
    field_str = f"{field:04d}"
    run_dir = str(run)

    try:
        # 一次性加载 5 个波段
        for band in BANDS:
            filename = f"frame-{band}-{run_str}-{camcol}-{field_str}.fits.bz2"
            file_path = os.path.join(root_dir, run_dir, str(camcol), filename)

            if not os.path.exists(file_path):
                # 如果某个波段文件缺失，这个 Field 下的所有对象都无法处理，直接放弃
                return processed_count, edge_count, error_ids, edge_ids, f"Missing file: {filename}"

            # 解压并读取
            with bz2.BZ2File(file_path, 'rb') as f_bz2:
                # 这里的 IO 和解压只发生一次！
                with fits.open(BytesIO(f_bz2.read())) as hdul:
                    img_data = hdul[0].data.astype(np.float32)  # 转为float32节省内存
                    header = hdul[0].header
                    wcs = WCS(header)
                    loaded_bands[band] = (wcs, img_data)

        # 处理待处理列表 (objects_to_process)
        for obj in objects_to_process:
            obj_id = obj['objID']
            save_path = os.path.join(save_dir, f"{obj_id}.npy")

            # 双重检查：防止极低概率的多进程写入冲突（可选）
            if os.path.exists(save_path):
                continue  # 已存在跳过

            try:
                ra, dec = obj['ra'], obj['dec']
                image_stack = np.zeros((5, CROP_SIZE, CROP_SIZE), dtype=np.float32)
                is_obj_edge = False

                # 对 5 个波段切片
                for i, band in enumerate(BANDS):
                    wcs, img_data = loaded_bands[band]

                    # 坐标转换 (纯内存计算，极快)
                    x_float, y_float = wcs.wcs_world2pix(ra, dec, 0)
                    x_c, y_c = int(np.round(x_float)), int(np.round(y_float))

                    h, w = img_data.shape

                    # 计算裁剪坐标
                    x1, x2 = x_c - HALF_SIZE, x_c + HALF_SIZE
                    y1, y2 = y_c - HALF_SIZE, y_c + HALF_SIZE
                    tx1, tx2, ty1, ty2 = 0, CROP_SIZE, 0, CROP_SIZE

                    # 边界处理
                    if x1 < 0: tx1, x1 = -x1, 0
                    if y1 < 0: ty1, y1 = -y1, 0
                    if x2 > w: tx2, x2 = CROP_SIZE - (x2 - w), w
                    if y2 > h: ty2, y2 = CROP_SIZE - (y2 - h), h

                    # 判定边缘
                    if (tx1 > 0) or (ty1 > 0) or (tx2 < CROP_SIZE) or (ty2 < CROP_SIZE):
                        is_obj_edge = True

                    # 内存复制
                    if tx2 > tx1 and ty2 > ty1:
                        image_stack[i, ty1:ty2, tx1:tx2] = img_data[y1:y2, x1:x2]

                # 保存
                np.save(save_path, image_stack)
                processed_count += 1
                if is_obj_edge:
                    edge_count += 1
                    edge_ids.append(obj_id)

            except Exception as e:
                error_ids.append((obj_id, str(e)))
                continue

        return processed_count, edge_count, error_ids, edge_ids, None

    except Exception as e:
        # 如果加载大图失败
        return 0, 0, [], [], f"Field Load Error: {str(e)}"


# ==========================================
# Worker: 光谱处理 (重采样与插值版)
# ==========================================
def worker_process_spectrum(args):
    """
    单个光谱处理任务：读取并重采样到统一对数波长网格
    args: (row_dict, save_path, spec_root)
    注意：target_grid 不作为参数传递，直接在函数内生成或作为全局变量使用以减少pickle开销
    """
    row, save_path, root_dir = args
    obj_id = row['objID']

    if os.path.exists(save_path):
        return 0, obj_id, "Skipped"

    try:
        plate, mjd, fiberid = int(row['plate']), int(row['mjd']), int(row['fiberID'])
        filename = f"spec-{plate:04d}-{mjd:05d}-{fiberid:04d}.fits"
        file_path = os.path.join(root_dir, f"{plate:04d}", filename)

        if not os.path.exists(file_path):
            return 3, obj_id, "File not found"

        with fits.open(file_path) as hdul:
            data = hdul[1].data
            # SDSS lite 数据通常包含 loglam (log10 wavelength)
            flux = data['flux'].astype(np.float32)
            loglam = data['loglam'].astype(np.float32)
            ivar = data['ivar'].astype(np.float32)  # 读取噪声反方差

            # 【关键步骤】基于 SDSS 标准的清洗
            # 1. 找到坏点：ivar 为 0 (无效) 或 ivar 极小 (噪声极大)
            # 2. 简单的策略是将坏点通量设为 0，或者设为整条光谱的中值
            #    对于深度学习，设为 0 是比较安全的，类似于 padding
            bad_mask = (ivar <= 1e-4) | (~np.isfinite(flux))
            flux[bad_mask] = 0.0

        # ==========================================
        # 核心逻辑: 对数波长重采样
        # ==========================================
        # 1. 生成目标网格 (Log-Linear Grid)
        # 范围: log10(3800) ~ log10(9000), 点数: 4096
        target_log_grid = np.linspace(np.log10(WAVE_START), np.log10(WAVE_END), TARGET_LEN)

        # 2. 线性插值 (Linear Interpolation)
        # np.interp(x_target, x_source, y_source, left=0, right=0)
        # 这里的 left=0, right=0 实现了"超出部分自动填0"
        resampled_flux = np.interp(target_log_grid, loglam, flux, left=0.0, right=0.0)

        # 3. 转换为 float32 并在保存前确保形状正确
        final_spec = resampled_flux.astype(np.float32)

        np.save(save_path, final_spec)

        # 状态码: 0=Success
        return 0, obj_id, "Success"

    except Exception as e:
        return 4, obj_id, str(e)


# ==========================================
# 主 Pipeline 类
# ==========================================
class AstroPipeline:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到星表文件: {csv_path}")

        print(f"正在读取 CSV: {csv_path} ...")
        self.df = pd.read_csv(csv_path)
        self.df['objID'] = self.df['objID'].astype(str)

        self.img_dir = os.path.join(CACHE_DIR, 'images')
        self.spec_dir = os.path.join(CACHE_DIR, 'spectra')
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.spec_dir, exist_ok=True)

    def process_images_optimized(self):
        print(f"正在进行图像任务分组 (Group by Field)...")

        # 1. 按照 (run, camcol, field) 对任务进行分组
        # 这样我们可以一次 IO，处理多个对象
        field_groups = defaultdict(list)

        # 只选取需要的列以加快循环
        subset = self.df[['objID', 'run', 'camcol', 'field', 'ra', 'dec']].to_dict('records')

        for row in subset:
            key = (row['run'], row['camcol'], row['field'])
            field_groups[key].append({
                'objID': str(row['objID']),
                'ra': row['ra'],
                'dec': row['dec']
            })

        tasks = []
        for (run, camcol, field), obj_list in field_groups.items():
            tasks.append((run, camcol, field, obj_list, IMAGE_ROOT, self.img_dir))

        print(f"共 {len(self.df)} 个对象，合并为 {len(tasks)} 个 Field 组任务。")
        print(f"开始并行处理图像 (Workers={NUM_WORKERS})...")

        total_success = 0
        total_edge = 0
        all_edge_ids = []

        # 并行执行
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # chunksize=1 保证任务分配均匀
            results = list(tqdm(executor.map(worker_process_field_group, tasks, chunksize=1),
                                total=len(tasks), desc="Processing Fields"))

        # 统计结果
        for proc_count, edge_count, error_ids, edge_ids, group_err in results:
            if group_err:
                print(f"[Group Error] {group_err}")
                continue

            total_success += proc_count
            total_edge += edge_count
            all_edge_ids.extend(edge_ids)

            for oid, err in error_ids:
                # 这里的错误通常是 WCS 转换越界严重等单体错误
                pass

        # 写入日志
        log_file = os.path.join(CACHE_DIR, 'edge_cases_log.txt')
        with open(log_file, 'a') as f:
            # 写入时间
            f.write(f"{datetime.now()}\n")
            f.write(f"Edge Case Log - Total: {len(all_edge_ids)}\n")
            for eid in all_edge_ids: f.write(f"{eid}\n")

        print("-" * 40)
        print(f"图像处理(极速版)完成:")
        print(f"成功保存对象数: {total_success}")
        print(f"边缘填充对象数: {total_edge}")
        print("-" * 40)

    def process_spectra(self):
        print(f"开始并行处理光谱数据 (Workers={NUM_WORKERS})...")
        print(f"重采样配置: {WAVE_START}-{WAVE_END} A (log space), Length={TARGET_LEN}")

        tasks = []
        for _, row in self.df.iterrows():
            row_dict = {'objID': str(row['objID']), 'plate': row['plate'], 'mjd': row['mjd'], 'fiberID': row['fiberID']}
            # 注意：不再传递 TARGET_LEN，因为已经在 worker 内部使用全局常量或重新计算
            tasks.append((row_dict, os.path.join(self.spec_dir, f"{row_dict['objID']}.npy"), SPEC_ROOT))

        success_cnt = 0
        fail_cnt = 0
        log_missing = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(
                tqdm(executor.map(worker_process_spectrum, tasks, chunksize=100), total=len(tasks), desc="Spectra"))

        for status, obj_id, msg in results:
            if status == 0:
                success_cnt += 1
            else:
                fail_cnt += 1
                if status == 3:
                    log_missing.append(f"{obj_id} (Missing)")
                elif status == 4:
                    log_missing.append(f"{obj_id} (Error: {msg})")

        if log_missing:
            with open(os.path.join(CACHE_DIR, 'spec_missing_log.txt'), 'a') as f:
                for item in log_missing: f.write(f"{item}\n")

        print("-" * 40)
        print(f"光谱处理完成。")
        print(f"成功: {success_cnt}")
        print(f"失败: {fail_cnt} (详情见 spec_missing_log.txt)")
        print("-" * 40)

    def assemble_dataset(self):
        # 保持原有 assemble 逻辑，无需改动
        print("开始组装 HF Dataset...")
        features = Features({
            'objID': Value('string'),
            'image': Array3D(shape=(5, 64, 64), dtype='float32'),
            'spectrum': Sequence(feature=Value('float32'), length=TARGET_LEN),
            'params': Sequence(feature=Value('float32'), length=15),
            'z': Value('float32')
        })

        data_records = self.df.to_dict('records')
        img_dir_base, spec_dir_base = self.img_dir, self.spec_dir

        def gen():
            missing_count = 0
            for row in data_records:
                obj_id = str(row['objID'])
                img_path = os.path.join(img_dir_base, f"{obj_id}.npy")
                spec_path = os.path.join(spec_dir_base, f"{obj_id}.npy")

                # 简单检查，不使用 os.path.exists 以极致提速 (利用 try-except)
                try:
                    img_data = np.load(img_path)
                    spec_data = np.load(spec_path)

                    # 检查光谱形状，确保符合 4096
                    if spec_data.shape[0] != TARGET_LEN:
                        # 理论上不会发生，除非有旧缓存文件
                        raise ValueError(f"Shape mismatch: {spec_data.shape}")

                    yield {
                        'objID': obj_id,
                        'image': img_data,
                        'spectrum': spec_data,
                        'params': np.array([row[c] for c in PARAMS_COLS], dtype=np.float32),
                        'z': float(row['z'])
                    }
                except (FileNotFoundError, OSError):
                    missing_count += 1
                    continue
            print(f"跳过缺失数据: {missing_count}")

        ds = Dataset.from_generator(gen, features=features)
        print(f"总样本数: {len(ds)}")
        ds_split = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
        ds_split['train'].save_to_disk(os.path.join(OUTPUT_DIR, 'train_dataset'))
        ds_split['test'].save_to_disk(os.path.join(OUTPUT_DIR, 'test_dataset'))
        print("全部完成！")


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--step_image', action='store_true')
    group.add_argument('--step_spec', action='store_true')
    group.add_argument('--step_assemble', action='store_true')
    group.add_argument('--run_all', action='store_true')
    args = parser.parse_args()

    pipeline = AstroPipeline(CSV_FILE)

    if args.step_image or args.run_all:
        pipeline.process_images_optimized()  # 调用新的优化版函数
    if args.step_spec or args.run_all:
        pipeline.process_spectra()
    if args.step_assemble or args.run_all:
        pipeline.assemble_dataset()
