import numpy as np
import pandas as pd
from astropy.io import fits
import os

def check_fits_objid(fits_path, hdu_index=1):
    """
    检测 FITS 文件中 OBJID 字段的重复和空值情况。
    """
    if not os.path.exists(fits_path):
        print(f"❌ 错误: 找不到文件 {fits_path}")
        return

    print(f"📂 正在读取文件: {fits_path} (HDU: {hdu_index})...")
    
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[hdu_index].data
        
        # 检查是否存在 OBJID 列
        if 'OBJID' not in data.columns.names:
            print("❌ 错误: FITS 文件中不存在 'OBJID' 列！")
            return
            
        print(f"✅ 成功读取数据，总行数: {len(data)}")
        
        # 提取 OBJID 列
        raw_objid = data['OBJID']
        
        # 统一清洗格式：处理字节串(bytes)和去除尾部无用空格
        print("🧹 正在清洗和格式化 OBJID 数据...")
        if raw_objid.dtype.kind in ('S', 'O'):  # 如果是字节串或对象类型
            cleaned_objids = [
                x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip() 
                for x in raw_objid
            ]
        else:
            cleaned_objids = [str(x).strip() for x in raw_objid.tolist()]

    # 将清洗后的数据转为 Pandas Series 以便快速统计
    objid_series = pd.Series(cleaned_objids)
    
    # 统计所有值的出现频率
    value_counts = objid_series.value_counts()
    
    print("\n" + "="*50)
    print("📊 统计结果报告")
    print("="*50)
    
    # 1. 检测“空值”和“无效值” (在SDSS中通常是 '', '0', '-1')
    invalid_markers = ['', '0', '-1']
    print("【空值/无效值检测】")
    found_invalid = False
    for marker in invalid_markers:
        if marker in value_counts:
            display_marker = "空字符串 ''" if marker == '' else f"'{marker}'"
            print(f"  ⚠️ 发现无效值 {display_marker}: 出现了 {value_counts[marker]} 次")
            found_invalid = True
            
    if not found_invalid:
        print("  ✅ 未发现常见的无效或空 OBJID ('', '0', '-1')。")

    print("-" * 50)
    
    # 2. 检测常规的重复项 (排除我们刚才已经统计过的无效值)
    print("【真实重复项检测 (排除无效值)】")
    
    # 过滤掉无效值
    valid_counts = value_counts[~value_counts.index.isin(invalid_markers)]
    
    # 找出出现次数大于 1 的项
    duplicates = valid_counts[valid_counts > 1]
    
    if duplicates.empty:
        print("  ✅ 恭喜！除了可能的无效值外，没有发现任何常规 OBJID 重复。")
    else:
        print(f"  ⚠️ 发现 {len(duplicates)} 个不同的有效 OBJID 存在重复现象！")
        print("\n  重复次数最多的前 15 个 OBJID 如下：")
        
        # 打印前 15 个最常重复的 OBJID
        top_duplicates = duplicates.head(15)
        for objid, count in top_duplicates.items():
            print(f"    - OBJID: {objid:<25} | 重复次数: {count}")
            
        # 如果重复项太多，可以选择导出到 CSV
        if len(duplicates) > 15:
            output_dup_file = "duplicate_objids_report.csv"
            duplicates.to_csv(output_dup_file, header=['Count'])
            print(f"\n  📝 注意: 还有更多重复项未显示，完整重复列表已保存至: {output_dup_file}")

    print("="*50)

# ==========================================
# 执行部分
# ==========================================
if __name__ == "__main__":
    # 将这里替换为你的 FITS 文件路径
    INPUT_FITS_FILE = "dsm/fits/DR16Q_v4.fits"  
    check_fits_objid(INPUT_FITS_FILE)