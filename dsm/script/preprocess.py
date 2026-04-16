import pandas as pd
import os

def explore_dataset_fields(csv_path):
    """
    快速读取 CSV 文件的表头，输出所有字段名。
    使用 nrows=0 避免加载全量数据，提高执行效率。
    """
    print(f"📂 正在检查文件: {csv_path} ...")
    
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到文件 {csv_path}")
        return

    # 仅读取表头
    df_header = pd.read_csv(csv_path, nrows=0)
    columns = df_header.columns.tolist()
    
    print("\n" + "="*50)
    print(f"📊 数据集字段概览 (共 {len(columns)} 个字段)")
    print("="*50)
    
    # 分类打印，让输出更易读
    print("【标量与标识符字段】")
    scalar_cols = [col for col in columns if not any(band in col[-2:] for band in ['_u', '_g', '_r', '_i', '_z'])]
    for col in scalar_cols:
        print(f"  - {col}")
        
    print("\n【多维测光字段 (u, g, r, i, z)】")
    # 找出带有波段后缀的列并提取前缀
    band_cols = [col for col in columns if col not in scalar_cols]
    prefixes = sorted(list(set([col.rsplit('_', 1)[0] for col in band_cols])))
    
    for prefix in prefixes:
        print(f"  - {prefix}_[u, g, r, i, z]")
        
    print("="*50)
    print("✅ 字段读取完毕！")

def analyze_data_distribution(csv_path):
    """
    分析 z、extinction、psfMag、psfMagErr 字段的统计信息。
    包括 mean, std, 最值(min/max), 百分位(1%, 5%, 95%, 99%)
    """
    print(f"\n📂 正在加载数据进行统计分析: {csv_path} ...")
    
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到文件 {csv_path}")
        return

    # 为了安全，依然指定 objID 为字符串，尽管这里我们不分析它
    df = pd.read_csv(csv_path, dtype={'objID': str})
    
    # 1. 动态生成需要分析的目标列名
    bands = ['u', 'g', 'r', 'i', 'z']
    target_cols = ['z'] # 红移
    for prefix in ['extinction', 'psfMag', 'psfMagErr']:
        for band in bands:
            target_cols.append(f"{prefix}_{band}")
            
    # 过滤掉数据集中可能不存在的列 (做个安全校验)
    valid_cols = [col for col in target_cols if col in df.columns]
    
    print(f"✅ 数据加载完成！正在计算 {len(valid_cols)} 个特征的统计指标...")
    
    # 2. 计算统计量
    # 自定义需要的百分位点
    percentiles = [0.01, 0.05, 0.95, 0.99]
    stats = df[valid_cols].describe(percentiles=percentiles)
    
    # describe() 默认会包含 'count' 和 '50%' (中位数)，我们挑选出你需要的指标
    display_metrics = ['mean', 'std', 'min', '1%', '5%', '95%', '99%', 'max']
    
    # 将 DataFrame 转置 (.T)，让特征在行，指标在列，这样控制台打印出来更清晰易读
    stats_display = stats.loc[display_metrics].T
    
    # 3. 打印美观的报告
    print("\n" + "="*85)
    print("📊 关键字段统计分析报告")
    print("="*85)
    
    # 临时调整 pandas 打印设置，防止列被折叠，并保留 4 位小数
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: f"{x:10.4f}")
    
    print(stats_display)
    
    # 恢复 pandas 默认设置
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    print("="*85)
    
    # 4. 可选：将统计结果导出，方便用 Excel 慢慢查看
    output_stats_file = csv_path.replace(".csv", "_statistics_report.csv")
    stats_display.to_csv(output_stats_file)
    print(f"💾 完整的统计报告已保存至: {output_stats_file}")

def clean_missing_values(input_csv, output_csv):
    """
    清洗无效观测数据：删除包含 -999 / -9999 等标记缺失值的行。
    """
    print(f"📂 正在读取数据进行清洗: {input_csv} ...")
    
    if not os.path.exists(input_csv):
        print(f"❌ 错误: 找不到文件 {input_csv}")
        return

    # 强制指定 objID 为字符串类型读取
    df = pd.read_csv(input_csv, dtype={'objID': str})
    initial_count = len(df)
    print(f"✅ 成功读取数据，初始行数: {initial_count}")

    # ==========================================
    # 核心清洗逻辑：寻找并标记异常值
    # ==========================================
    # 获取所有的数值类型列（排除 objID 等字符串字段）
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # 创建一个与数据行数相同的布尔序列，初始全为 False（代表不是异常行）
    invalid_mask = pd.Series(False, index=df.index)
    
    # 逐列检查：只要某一行在任意一个数值列中出现了 <= -900 的极值，就判定该行为无效观测
    for col in numeric_cols:
        col_invalid = df[col] <= -900
        invalid_mask = invalid_mask | col_invalid
        
    # 提取过滤后的干净数据 (~ 是取反操作，保留不含异常的行)
    clean_df = df[~invalid_mask]
    
    # 统计数据
    dropped_count = invalid_mask.sum()
    remaining_count = len(clean_df)
    
    # ==========================================
    # 打印处理报告
    # ==========================================
    print("\n" + "="*55)
    print("🧹 数据清洗报告 (清理 -999 无效观测)")
    print("="*55)
    print(f"  🔹 原始数据量:   {initial_count} 条")
    print(f"  🔻 删除异常数据: {dropped_count} 条")
    print(f"  ✅ 剩余数据量:   {remaining_count} 条")
    
    if dropped_count > 0:
        print(f"  ▪️ 剔除率:       {(dropped_count / initial_count):.2%}")
    print("="*55)

    # ==========================================
    # 导出清洗后的数据
    # ==========================================
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    print(f"\n💾 第一阶段清洗完成！数据已保存至: {output_csv}")

def clean_outliers(input_csv, output_csv):
    """
    清洗物理异常值与离群点：
    1. 删除 z < 0 的数据 (不符合宇宙学红移物理意义)
    2. 删除 psfMagErr_[u,g,r,i,z] 大于各自 99% 百分位点的数据 (剔除观测误差极大的噪声)
    """
    print(f"\n📂 正在读取数据进行异常值清理: {input_csv} ...")
    
    if not os.path.exists(input_csv):
        print(f"❌ 错误: 找不到文件 {input_csv}")
        return

    # 保护 objID 精度，强制按字符串读取
    df = pd.read_csv(input_csv, dtype={'objID': str})
    initial_count = len(df)
    print(f"✅ 成功读取数据，参与此轮清理的初始行数: {initial_count}")

    # ==========================================
    # 1. 处理 z 字段 (红移异常)
    # ==========================================
    mask_z = df['z'] < 0
    z_dropped_count = mask_z.sum()

    # ==========================================
    # 2. 处理 psfMagErr 字段 (观测误差极值)
    # ==========================================
    # 自动找出所有的 psfMagErr 列 (u, g, r, i, z)
    err_cols = [col for col in df.columns if col.startswith('psfMagErr_')]
    
    # 初始化一个全 False 的布尔序列
    mask_err = pd.Series(False, index=df.index)
    percentile_99_dict = {}

    # 遍历每个波段，分别计算它的 99% 阈值
    for col in err_cols:
        p99 = df[col].quantile(0.99)
        percentile_99_dict[col] = p99
        # 只要该行在任意一个波段的误差大于该波段的 99% 阈值，就标记为 True (需要剔除)
        mask_err = mask_err | (df[col] > p99)

    err_dropped_count = mask_err.sum()

    # ==========================================
    # 3. 综合掩码并执行剔除
    # ==========================================
    # 只要满足上述任一条件，就进行剔除 (使用逻辑或 '|')
    mask_all = mask_z | mask_err
    total_dropped = mask_all.sum()
    
    # 提取保留下来的干净数据 (~ 是取反)
    clean_df = df[~mask_all]
    remaining_count = len(clean_df)

    # ==========================================
    # 4. 打印详细处理报告
    # ==========================================
    print("\n" + "="*60)
    print("🧹 第二阶段清洗报告 (异常值与离群点剔除)")
    print("="*60)
    print(f"  🔹 初始数据量:                 {initial_count} 条")
    print(f"  🔻 触发红移 z < 0 规则:        {z_dropped_count} 条")
    print(f"  🔻 触发误差 > 99% 阈值规则:    {err_dropped_count} 条")
    print("-" * 60)
    print(f"  🧨 综合实际剔除数 (去除交集):  {total_dropped} 条")
    print(f"  ✅ 剩余干净数据量:             {remaining_count} 条")
    print("-" * 60)
    
    print("  [附录] psfMagErr 各波段 99% 截断阈值参考:")
    for col, p99 in percentile_99_dict.items():
        print(f"       - {col}: {p99:.5f}")
    print("="*60)

    # ==========================================
    # 5. 保存为新 CSV
    # ==========================================
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    print(f"\n💾 第二阶段清洗完成！数据已保存至: {output_csv}")

def filter_high_redshift(input_csv, output_csv):
    """
    剔除极端高红移数据：
    删除 z > 6 的类星体，保留 z <= 6 的常规光学/近紫外类星体样本。
    """
    print(f"\n📂 正在读取数据进行高红移截断: {input_csv} ...")
    
    if not os.path.exists(input_csv):
        print(f"❌ 错误: 找不到文件 {input_csv}")
        return

    # 依然必须强制指定 objID 为字符串类型
    df = pd.read_csv(input_csv, dtype={'objID': str})
    initial_count = len(df)
    print(f"✅ 成功读取数据，参与此轮过滤的初始行数: {initial_count}")

    # ==========================================
    # 核心过滤逻辑
    # ==========================================
    # 找出所有 z > 6 的行
    mask_high_z = df['z'] > 6
    dropped_count = mask_high_z.sum()
    
    # 提取 z <= 6 的数据保存下来
    clean_df = df[~mask_high_z]
    remaining_count = len(clean_df)

    # ==========================================
    # 打印最终处理报告
    # ==========================================
    print("\n" + "="*60)
    print("✂️ 第三阶段清洗报告 (剔除极端高红移 z > 6)")
    print("="*60)
    print(f"  🔹 初始数据量:                 {initial_count} 条")
    print(f"  🔻 发现并剔除 z > 6 的数据量:  {dropped_count} 条")
    print("-" * 60)
    print(f"  ✅ 剩余高质量样本总量:         {remaining_count} 条")
    print("="*60)

    # ==========================================
    # 导出到新 CSV 文件
    # ==========================================
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    print(f"\n💾 第三阶段过滤完成！数据已保存至: {output_csv}")

if __name__ == "__main__":
    # 我们刚刚从 FITS 提取出来的纯净且无重复的起始数据集
    input_file = "dsm/script/DR16Q_v4_origin.csv"

    # 探索数据集字段
    # explore_dataset_fields(input_file)

    # 分析数据分布
    # analyze_data_distribution(input_file)

    # ==========================================
    # 第一阶段：执行清理 -999 数据的操作
    # ==========================================
    step1_csv = "dsm/script/DR16Q_v4_clean_step1.csv"
    # clean_missing_values(input_file, step1_csv)
    # 分析清洗后的数据分布
    # analyze_data_distribution(step1_csv)
    

    # ==========================================
    # 第二阶段：异常值与离群点剔除
    # ==========================================
    step2_csv = "dsm/script/DR16Q_v4_clean_step2.csv"
    # clean_outliers(step1_csv, step2_csv)
    # 分析清洗后的数据分布
    # analyze_data_distribution(step2_csv)

    # ==========================================
    # 第三阶段：高红移剔除
    # ==========================================
    step3_csv = "dsm/script/DR16Q_v4_clean_step3.csv"
    # filter_high_redshift(step2_csv, step3_csv)
    # 分析清洗后的数据分布
    # analyze_data_distribution(step3_csv)
    # explore_dataset_fields(step3_csv)

    # analyze_data_distribution("dsm/csv/DR16Q_v4.csv")


