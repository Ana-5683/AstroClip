import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from astroclip import format_with_env
from models import zero_shot,few_shot
def load_data(train_path, test_path, model_name):
    """加载训练和测试的嵌入及红移数据"""
    try:
        train_data = np.load(train_path)
        test_data = np.load(test_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。")
        print("请确保您已经运行了 get_embedding.py 脚本来生成嵌入文件。")
        sys.exit(1)  # 错误退出

    embedding_key = f"{model_name}_embeddings"
    if embedding_key not in train_data or embedding_key not in test_data:
        print(f"错误: 在 .npz 文件中找不到键 '{embedding_key}'。")
        print(f"请确保 --model_name 参数 ('{model_name}') 与生成嵌入时使用的模型名称一致。")
        sys.exit(1)  # 错误退出

    X_train = train_data[embedding_key]
    y_train = train_data['z']
    X_test = test_data[embedding_key]
    y_test = test_data['z']

    print(f"数据加载成功。训练集样本数: {len(X_train)}，测试集样本数: {len(X_test)}")
    return X_train, y_train, X_test, y_test


def evaluate_and_plot(y_true, y_pred, model_title, output_dir, log_file):
    """计算指标、绘制预测结果图并记录日志"""
    # 1. 计算 R2 分数
    r2 = r2_score(y_true, y_pred)

    # 2. 计算相对偏差
    delta_z = np.abs(y_pred - y_true) / (1 + y_true)

    # 3. 计算小于阈值的比例
    frac_01 = np.mean(delta_z < 0.1) * 100
    frac_02 = np.mean(delta_z < 0.2) * 100

    # 4. 新增：计算相对误差的 RMS 分数
    rms_relative = np.sqrt(np.mean(delta_z ** 2))

    # 准备要输出的文本
    header = f"--- 评估结果: {model_title} ---"
    r2_text = f"  R² 分数: {r2:.4f}"
    rms_text = f"  相对误差 RMS (σ_Δz/(1+z)): {rms_relative:.4f}"
    frac_01_text = f"  相对偏差 < 0.1 的比例: {frac_01:.2f}%"
    frac_02_text = f"  相对偏差 < 0.2 的比例: {frac_02:.2f}%"

    # 打印到控制台
    print(header)
    print(r2_text)
    print(rms_text)
    print(frac_01_text)
    print(frac_02_text)

    # 写入日志文件
    log_file.write(header + '\n')
    log_file.write(r2_text + '\n')
    log_file.write(rms_text + '\n')
    log_file.write(frac_01_text + '\n')
    log_file.write(frac_02_text + '\n\n')  # 添加一个空行以分隔不同模型的结果

    # 4. 绘制图像 (这部分逻辑不变)
    plt.figure(figsize=(8, 8))
    # sample_indices = np.random.choice(len(y_true), size=min(5000, len(y_true)), replace=False)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="完美预测 (y=x)")

    # 在标题中显示 R² 和 RMS
    plt.title(f"{model_title} - Redshift prediction results\n"
              f"$R^2={r2:.4f}$, $\\sigma_{{\\Delta z/(1+z)}}={rms_relative:.4f}$")
    plt.xlabel("z_true")
    plt.ylabel("z_pred")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()

    output_path = os.path.join(output_dir,f"{model_title}_redshift_prediction.png")
    plt.savefig(output_path)
    print(f"  预测结果图已保存至: {output_path}")
    plt.close()


def visualize_with_tsne(embeddings, redshifts, model_name, output_dir):
    """使用 t-SNE 对嵌入进行可视化"""
    print("\n正在进行 t-SNE 降维可视化...")
    # n_samples = min(5000, len(embeddings))
    # sample_indices = np.random.choice(len(embeddings), size=n_samples, replace=False)
    # sampled_embeddings = embeddings[sample_indices]
    # sampled_redshifts = redshifts[sample_indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=redshifts, cmap='viridis', s=10)

    plt.title(f"t-SNE visualization ( {model_name} )")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("真实红移 (z_true)")
    plt.grid(True)

    output_path = os.path.join(output_dir, f"{model_name}_tsne_visualization.png")
    plt.savefig(output_path)
    print(f"t-SNE 可视化图像已保存至: {output_path}")
    plt.close()


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")

    parser = argparse.ArgumentParser(description="使用嵌入进行红移预测")
    parser.add_argument("--model", type=str, required=True,
                        choices=["astroclip_image", "astroclip_spectrum","astroclip_photo","astroclip_ip" ,"astrodino", "specformer"],
                        help="用于生成嵌入的模型名称。")
    parser.add_argument("--pretrained_dir", type=str, default=f"{ASTROCLIP_ROOT}/pretrained/embeddings",
                        help="包含嵌入文件的目录。")
    parser.add_argument("--output_dir", type=str, default=f"{ASTROCLIP_ROOT}/pretrained/results", help="保存图表和结果的目录。")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 定义日志文件路径
    log_file_path = os.path.join(args.output_dir, "log.txt")

    # 使用 'with' 语句来安全地打开和写入文件
    with open(log_file_path, 'w') as log_file:
        print(f"日志将保存到: {log_file_path}")

        # 写入日志文件的标题
        log_file.write(f"--- 针对 '{args.model}' 嵌入的红移预测日志 ---\n\n")

        train_embedding_path = os.path.join(args.pretrained_dir, f"train_{args.model}_embedding.npz")
        test_embedding_path = os.path.join(args.pretrained_dir, f"test_{args.model}_embedding.npz")

        X_train, y_train, X_test, y_test = load_data(train_embedding_path, test_embedding_path, args.model)

        # --- 标准化 ---
        # 1. 对输入特征 X 进行标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # 2. 对目标标签 y (红移) 进行标准化
        scaler_y = StandardScaler()
        # StandardScaler 需要 2D 输入，所以我们 reshape y_train
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        # 将 y_train_scaled 变回 1D 数组以传递给模型
        y_train_scaled = y_train_scaled.ravel()

        # --- 调用模型进行训练和评估 ---

        # 1. 调用 few_shot 函数 (MLP)
        print("\n--- 正在使用 few_shot (MLP) 模型 ---")
        y_pred_mlp_scaled = few_shot(X_train_scaled, y_train_scaled, X_test_scaled)
        # 将预测值从标准化尺度还原到原始红移尺度
        y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1))
        y_pred_mlp = y_pred_mlp.ravel() # 变回 1D 数组用于评估
        # 使用还原后的预测值和原始的 y_test 进行评估
        evaluate_and_plot(y_test, y_pred_mlp, f"{args.model}_MLP", args.output_dir, log_file)

        # 2. 调用 zero_shot 函数 (KNN)
        print("\n--- 正在使用 zero_shot (KNN) 模型 ---")
        y_pred_knn_scaled  = zero_shot(X_train_scaled, y_train_scaled, X_test_scaled, n_neighbors=15)  # 可以调整 n_neighbors
        # 将预测值从标准化尺度还原到原始红移尺度
        y_pred_knn = scaler_y.inverse_transform(y_pred_knn_scaled.reshape(-1, 1))
        y_pred_knn = y_pred_knn.ravel()
        # 使用还原后的预测值和原始的 y_test 进行评估
        evaluate_and_plot(y_test, y_pred_knn, f"{args.model}_KNN", args.output_dir, log_file)

        # 3. t-SNE 可视化 (保持不变)
        visualize_with_tsne(X_test_scaled, y_test, args.model, args.output_dir)

    print(f"\n所有任务完成。结果已保存至 '{args.output_dir}' 目录。")
