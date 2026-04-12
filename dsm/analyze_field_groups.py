import pandas as pd
import os
import argparse


def analyze_groups(csv_path):
    print(f"正在读取文件: {csv_path} ...")
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    # 为了加快读取速度，我们只读取分组需要的列
    use_cols = ['objID', 'run', 'camcol', 'field']

    try:
        df = pd.read_csv(csv_path, usecols=use_cols)
    except ValueError:
        print("警告: 列名匹配失败，尝试读取所有列...")
        df = pd.read_csv(csv_path)

    total_objects = len(df)
    print(f"数据总行数 (Objects): {total_objects}")

    print("正在进行分组 (Group by run, camcol, field)...")

    # 执行分组统计
    # size() 返回每个组的大小
    group_counts = df.groupby(['run', 'camcol', 'field']).size()

    total_groups = len(group_counts)

    # 计算统计指标
    max_in_group = group_counts.max()
    min_in_group = group_counts.min()
    mean_in_group = group_counts.mean()
    median_in_group = group_counts.median()

    # 计算理论提升倍数
    # 旧方案操作次数 = total_objects
    # 新方案操作次数 = total_groups
    speedup_ratio = total_objects / total_groups

    print("=" * 40)
    print("分组统计结果")
    print("=" * 40)
    print(f"1. 视场 (Fields) 总数 : {total_groups}")
    print(f"   (这意味着原本要解压 {total_objects} 次，现在只需解压 {total_groups} 次)")
    print("-" * 40)
    print(f"2. 每个视场包含的目标数统计:")
    print(f"   - 平均值 (Mean)   : {mean_in_group:.2f} 个/视场")
    print(f"   - 中位数 (Median) : {median_in_group:.0f} 个/视场")
    print(f"   - 最大值 (Max)    : {max_in_group} 个/视场")
    print(f"   - 最小值 (Min)    : {min_in_group} 个/视场")
    print("-" * 40)
    print(f"3. 理论 IO 速度提升倍数 : {speedup_ratio:.2f} 倍")
    print("=" * 40)

    # ------------------------------------------------
    # 额外分析：分布详情
    # ------------------------------------------------
    print("\n[分布详情] 包含 N 个目标的视场有多少个？")
    print("(例如: 有多少个视场里只有 1 个目标，有多少个视场里有 10 个目标)")

    # 统计 "每个组的大小" 的出现频率
    # value_counts() 统计 count 的分布
    distribution = group_counts.value_counts().sort_index()

    # 打印前 10 种情况
    print(f"{'目标数/视场':<15} | {'视场数量':<10} | {'占比':<10}")
    print("-" * 40)

    count_printed = 0
    cum_percentage = 0

    for num_objs, num_fields in distribution.items():
        percentage = (num_fields / total_groups) * 100
        cum_percentage += percentage
        print(f"{num_objs:<15} | {num_fields:<10} | {percentage:.2f}%")

        count_printed += 1
        if count_printed >= 15:  # 只看前15行，后面太长不看
            print("... (更多分布略)")
            break

    # 打印最密集的视场示例（方便你去检查）
    print("\n[示例] 包含目标最多的视场 (Top 1):")
    top_group = group_counts.idxmax()  # 返回索引 (run, camcol, field)
    top_count = group_counts.max()
    print(f"Run: {top_group[0]}, Camcol: {top_group[1]}, Field: {top_group[2]} -> 包含 {top_count} 个类星体")


if __name__ == "__main__":
    # 你的 CSV 文件路径
    csv_file_path = './merged_all_clean.csv'

    analyze_groups(csv_file_path)