import pandas as pd
import os

def check_csv_objid(csv_path, output_clean_csv):
    print(f"📂 正在读取文件: {csv_path} ...")
    
    try:
        # 强制指定 objID 为字符串类型读取，防止精度丢失
        df = pd.read_csv(csv_path, dtype={'objID': str})
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_path}")
        return

    total_rows = len(df)
    print(f"✅ 成功读取数据，总行数: {total_rows}")

    # ==========================================
    # 1. 核心清洗逻辑
    # ==========================================
    # 找出 NaN / 空值
    is_null = df['objID'].isna() | (df['objID'] == '') | (df['objID'].str.lower() == 'nan')
    nan_count = is_null.sum()
    
    # 过滤出带有有效 objID 的数据
    valid_df = df[~is_null]

    # 统计重复的 OBJID 以及对应的个数
    value_counts = valid_df['objID'].value_counts()
    duplicates = value_counts[value_counts > 1]
    
    # 计算各项指标
    unique_valid_count = len(value_counts)       # 剩余独立有效目标数
    perfect_clean_count = (value_counts == 1).sum() # 绝对纯净数据量
    duplicate_rows_total = duplicates.sum()      # 参与重复的总行数

    # ==========================================
    # 2. 打印结果报告
    # ==========================================
    print("\n" + "="*55)
    print("📊 CSV 数据质量检验报告")
    print("="*55)
    print(f"🔹 原始数据总行数: {total_rows}")
    print("-" * 55)
    
    print(f"【1】空值检测")
    print(f"  ⚠️ objID 为 NaN 或空值的个数: {nan_count}")
    print("-" * 55)
    
    print(f"【2】重复项检测 (已排除 NaN)")
    if duplicates.empty:
        print("  ✅ 恭喜！有效的 objID 中没有任何重复项。")
    else:
        print(f"  ⚠️ 发现 {len(duplicates)} 个不同的 objID 存在重复现象 (共波及 {duplicate_rows_total} 行)。")
        # 只打印前 5 个，避免刷屏
        print("\n  重复详情列表 (前 5 个) 如下：")
        for objid, count in duplicates.head(5).items():
            print(f"    - objID: {objid:<22} | 出现次数: {count}")
            
    print("-" * 55)
    
    print(f"【3】剩余数据量核算")
    print(f"  ▪️ 去除 NaN 后的有效数据行数:      {len(valid_df)}")
    print(f"  ▪️ 剩余独立有效目标数 (保留1个重复项): {unique_valid_count}")
    print(f"  ▪️ 绝对纯净数据量 (剔除所有重复与NaN): {perfect_clean_count}")
    print("="*55)

    # ==========================================
    # 3. 导出最终可用数据
    # ==========================================
    print(f"\n💾 正在生成并保存去重后的最终数据集...")
    
    # 针对 objID 列去重，保留第一次出现的行 (keep='first')
    unique_df = valid_df.drop_duplicates(subset=['objID'], keep='first')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_clean_csv), exist_ok=True)
    
    # 写入新的 CSV，记得 index=False
    unique_df.to_csv(output_clean_csv, index=False)
    print(f"✅ 最终数据已成功保存至: {output_clean_csv}")
    print(f"🎉 导出的文件行数为: {len(unique_df)} 行 (应与上方的'剩余独立有效目标数'完全一致)")

if __name__ == "__main__":
    # 你刚刚生成的中间态文件
    csv_file = "dsm/script/sdss_quasar_dataset_2.csv"
    
    # 最终你要输出的干净文件路径 (建议放在同一个目录下)
    clean_output = "dsm/script/DR16Q_v4.csv" 
    
    check_csv_objid(csv_file, clean_output)