import os
import re


def collect_logs():
    # 1. 定义源目录和输出文件路径
    source_dir = "../pretrained/results"
    output_file = "./total_res.txt"
    target_filename = "astrodino_log.txt"

    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 目录 {source_dir} 不存在。")
        return

    # 2. 获取源目录下所有以 astrodino_27_ 开头的文件夹
    subfolders = []
    try:
        all_items = os.listdir(source_dir)
        for item in all_items:
            full_path = os.path.join(source_dir, item)
            if os.path.isdir(full_path) and item.startswith("astrodino_27_"):
                subfolders.append(item)
    except Exception as e:
        print(f"读取目录出错: {e}")
        return

    # 3. 对文件夹进行自然排序 (按数字大小，而不是字母顺序)
    # 例如确保 9999 排在 112499 前面
    def extract_number(folder_name):
        # 尝试提取文件夹名最后部分的数字
        parts = folder_name.split('_')
        if parts[-1].isdigit():
            return int(parts[-1])
        return 0  # 如果没有数字，默认排在最前

    subfolders.sort(key=extract_number)

    # 4. 遍历文件夹并收集内容
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for folder_name in subfolders:
            log_path = os.path.join(source_dir, folder_name, target_filename)

            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f_in:
                        content = f_in.read().strip()  # 读取并去除首尾多余空白

                        # 写入格式要求的 Header
                        f_out.write(f"{folder_name}: \n")
                        # 写入内容
                        f_out.write(content)
                        # 写入分隔符（换行），防止不同文件内容连在一起
                        f_out.write("\n\n" + "-" * 50 + "\n\n")

                        print(f"[收集成功] {folder_name}")
                        count += 1
                except Exception as e:
                    print(f"[读取错误] 无法读取 {folder_name} 中的日志: {e}")
            else:
                print(f"[文件缺失] {folder_name} 中没有 {target_filename}")

    print(f"\n汇总完成！共收集了 {count} 个日志文件。")
    print(f"结果已保存至: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    collect_logs()