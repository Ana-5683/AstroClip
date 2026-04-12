# evaluate_redshift.py

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Assuming your 'models' file with zero_shot and few_shot is available
# If not, we can define them here simply.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from astroclip import format_with_env
from models import zero_shot,few_shot

# --- The rest of the file is almost identical to your redshift.py ---

def load_data(train_path, test_path, model_name):
    try:
        train_data = np.load(train_path)
        test_data = np.load(test_path)
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}.")
        print("Please ensure you have run embed_photometry.py to generate embeddings.")
        sys.exit(1)

    embedding_key = f"{model_name}_embeddings"
    if embedding_key not in train_data or embedding_key not in test_data:
        print(f"Error: Key '{embedding_key}' not found in .npz files.")
        sys.exit(1)

    X_train = train_data[embedding_key]
    y_train = train_data['z']
    X_test = test_data[embedding_key]
    y_test = test_data['z']

    print(f"Data loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
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

    # --- 将结果同时打印到控制台和写入日志文件 ---

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

    parser = argparse.ArgumentParser(description="Evaluate redshift prediction from embeddings.")
    parser.add_argument("--model_name", type=str, default="photometry",
                        help="The name of the model embeddings to evaluate (e.g., 'photometry').")
    parser.add_argument("--ckpt", type=str, default=f"photoencoder_00")

    args = parser.parse_args()

    output_dir=os.path.join(f"{ASTROCLIP_ROOT}/pretrained/results",args.ckpt)
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, f"{args.model_name}_evaluation_log.txt")

    with open(log_file_path, 'w') as log_file:
        print(f"Logs will be saved to: {log_file_path}")
        log_file.write(f"--- Redshift Prediction Log for '{args.model_name}' Embeddings ---\n\n")
        embedding_dir=os.path.join(f"{ASTROCLIP_ROOT}/pretrained/embeddings",args.ckpt)

        train_path = os.path.join(embedding_dir, f"train_{args.model_name}_embedding.npz")
        test_path = os.path.join(embedding_dir, f"test_{args.model_name}_embedding.npz")

        X_train, y_train, X_test, y_test = load_data(train_path, test_path, args.model_name)

        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        print("\n--- Evaluating few_shot (MLP) model ---")
        y_pred_mlp_scaled = few_shot(X_train_scaled, y_train_scaled, X_test_scaled)
        y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).ravel()
        evaluate_and_plot(y_test, y_pred_mlp, f"{args.model_name}_MLP", output_dir, log_file)

        print("\n--- Evaluating zero_shot (KNN) model ---")
        y_pred_knn_scaled = zero_shot(X_train_scaled, y_train_scaled, X_test_scaled, n_neighbors=15)
        y_pred_knn = scaler_y.inverse_transform(y_pred_knn_scaled.reshape(-1, 1)).ravel()
        evaluate_and_plot(y_test, y_pred_knn, f"{args.model_name}_KNN", output_dir, log_file)

        visualize_with_tsne(X_test_scaled, y_test, args.model_name, output_dir)

    print(f"\nAll tasks complete. Results are in '{output_dir}'.")