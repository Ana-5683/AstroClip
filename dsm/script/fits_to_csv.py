import numpy as np
import pandas as pd
from astropy.io import fits
import os
import gc

def process_fits_in_chunks(fits_path, output_csv_path, chunk_size=50000):
    """
    分块读取 FITS 文件并转换 CSV，已修复多维数组大端序问题，
    并对 OBJID 进行了防空值、防字节串污染的强化处理。
    """
    if not os.path.exists(fits_path):
        print(f"❌ 错误: 找不到文件 {fits_path}")
        return

    # 标量列名
    scalar_cols = [
        'SDSS_NAME', 'THING_ID', 'OBJID', 'RA', 'DEC', 'PLATE', 'MJD', 
        'FIBERID', 'Z', 'RUN_NUMBER', 'CAMCOL_NUMBER', 'FIELD_NUMBER'
    ]
    
    # 多维矩阵列名
    multidim_cols = ['PSFMAG', 'PSFMAGERR', 'EXTINCTION']
    bands = ['u', 'g', 'r', 'i', 'z']

    print(f"🚀 开始处理 {fits_path} ...")

    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        total_rows = len(data)
        print(f"📊 总行数: {total_rows}")

        first_chunk = True

        # 开始分块循环
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            print(f"⏳ 正在处理第 {start_idx} 到 {end_idx} 行 (进度: {end_idx / total_rows:.1%})...")

            chunk_data = data[start_idx:end_idx]
            batch_dict = {}

            # ==========================================
            # 1. 处理标量列
            # ==========================================
            for col in scalar_cols:
                if col in data.columns.names:
                    # 【核心修复】强制解决 FITS 的 Big-Endian 问题
                    col_data = chunk_data[col].byteswap().newbyteorder()
                    
                    if col == 'OBJID':
                        safe_objids = []
                        # 遍历转换为 Python 原生类型
                        for x in col_data.tolist():
                            # 兼容解码 bytes 并去除多余空格
                            val = x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip()
                            
                            # 【应用决定 1A】将无效值转为空字符串，Pandas 读取时会自动识别为 NaN
                            if val in ['', '0', '-1']:
                                safe_objids.append('')
                            else:
                                # 【应用决定 2B】维持纯数字字符串原样
                                safe_objids.append(val)
                                
                        batch_dict[col] = safe_objids
                    else:
                        batch_dict[col] = col_data

            # ==========================================
            # 2. 处理多维列 (拆分 u, g, r, i, z)
            # ==========================================
            for col in multidim_cols:
                if col in data.columns.names:
                    # 【核心修复】多维矩阵列同样必须先进行 byteswap，否则浮点数会完全错乱
                    matrix = chunk_data[col].byteswap().newbyteorder()

                    for i, band in enumerate(bands):
                        batch_dict[f"{col}_{band}"] = matrix[:, i]

            # ==========================================
            # 3. 转换为 Pandas DataFrame 并格式化
            # ==========================================
            df_chunk = pd.DataFrame(batch_dict)
            
            # 列名全小写化
            df_chunk.columns = df_chunk.columns.str.lower()
            
            # 应用重命名映射字典
            rename_mapping = {
                'run_number': 'run', 'camcol_number': 'camcol', 'field_number': 'field',
                'objid': 'objID', 'fiberid': 'fiberID',
                'psfmag_u': 'psfMag_u', 'psfmag_g': 'psfMag_g', 'psfmag_r': 'psfMag_r', 
                'psfmag_i': 'psfMag_i', 'psfmag_z': 'psfMag_z',
                'psfmagerr_u': 'psfMagErr_u', 'psfmagerr_g': 'psfMagErr_g', 'psfmagerr_r': 'psfMagErr_r', 
                'psfmagerr_i': 'psfMagErr_i', 'psfmagerr_z': 'psfMagErr_z',
                'extinction_u': 'extinction_u', 'extinction_g': 'extinction_g', 'extinction_r': 'extinction_r',
                'extinction_i': 'extinction_i', 'extinction_z': 'extinction_z'
            }
            df_chunk.rename(columns=rename_mapping, inplace=True)

            # ==========================================
            # 4. 写入 CSV 与内存清理
            # ==========================================
            if first_chunk:
                df_chunk.to_csv(output_csv_path, mode='w', index=False, header=True)
                first_chunk = False
            else:
                df_chunk.to_csv(output_csv_path, mode='a', index=False, header=False)

            del df_chunk
            del batch_dict
            del chunk_data
            gc.collect()

    print(f"✅ 处理完成！文件已保存为: {output_csv_path}")


# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    input_fits = "dsm/fits/DR16Q_v4.fits"
    output_csv = "dsm/script/sdss_quasar_dataset_2.csv"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    process_fits_in_chunks(input_fits, output_csv, chunk_size=50000)