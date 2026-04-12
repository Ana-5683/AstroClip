# embed_photometry.py

import argparse
import os
import torch
import numpy as np
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from astroclip import format_with_env
from astroclip.photoencoder.data import PhotometryTransform
from astroclip.photoencoder.trainer import MaskedPhotometryModel


class InferencePhotometryDataset(Dataset):
    """
    专门用于推理的 Dataset。
    相比于训练时的 Dataset，它会额外返回 'z' (redshift) 信息用于保存。
    """

    def __init__(self, data_path: str, split: str = 'train', transform=None):
        if split == 'train':
            data_path = os.path.join(data_path, 'train_dataset')
        elif split == 'test':
            data_path = os.path.join(data_path, 'test_dataset')

        # 加载 HuggingFace 格式的磁盘数据集
        self.dataset = datasets.load_from_disk(data_path)
        self.transform = transform

        # 确保加载 params (测光) 和 z (红移)
        # 注意：这里假设你的 arrow/parquet 数据集中包含 'z' 这一列
        self.dataset.set_format(type="torch", columns=['params', 'z'])
        print(f"Loaded {len(self.dataset)} samples from {data_path} ({split})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        params_tensor = sample['params']  # Shape: (15,)
        z_tensor = sample['z']  # Shape: (1,) or scalar

        # 应用预处理 (15,) -> (5, 2)
        if self.transform:
            params_tensor = self.transform(params_tensor)

        return params_tensor, z_tensor


def generate_embeddings(
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """
    执行推理循环，生成 Embeddings 和 收集 Redshifts。
    """
    model.to(device)
    model.eval()

    all_embeddings = []
    all_redshifts = []

    with torch.no_grad():
        for batch_data, batch_z in tqdm(data_loader, desc="Generating Embeddings"):
            # batch_data: (B, 5, 2)
            batch_data = batch_data.to(device)

            # Forward pass
            # mask_indices=None 表示推理模式，不进行 Mask
            output = model(batch_data, mask_indices=None)

            # output['features'] shape: (B, 6, 128) -> [CLS, u, g, r, i, z]
            feats = output['features'][:,0]  # (B,128)

            # 转为 numpy 并收集
            all_embeddings.append(feats.cpu().numpy())
            all_redshifts.append(batch_z.numpy())

    # Concatenate all batches
    # Final Shape: (Total_Samples, 6, 128)
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_redshifts = np.concatenate(all_redshifts, axis=0)

    return final_embeddings, final_redshifts


def main(args):
    # 1. --- Setup Transform ---
    # 使用之前定义好的预处理类 (确保内部已填入计算好的 mean/std)
    transform = PhotometryTransform()

    # 2. --- Load Model from Checkpoint ---
    print(f"Loading model from checkpoint: {args.ckpt}")

    # 使用 PyTorch Lightning 的 load_from_checkpoint 自动处理参数加载
    # 这会自动加载超参数 (d_model, n_layers 等) 和 权重
    try:
        ckpt_path=os.path.join(ASTROCLIP_ROOT,"pretrained", f"{args.ckpt}.ckpt")
        pl_module = MaskedPhotometryModel.load_from_checkpoint(ckpt_path)
        # 我们只需要内部的 encoder 部分进行推理
        encoder = pl_module.model
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. --- Process Splits ---
    splits = ['train','test']

    for split in splits:
        print(f"\n--- Processing '{split}' dataset ---")

        # 初始化数据集
        try:
            dataset = InferencePhotometryDataset(
                data_path=args.data_path,
                split=split,
                transform=transform
            )
        except Exception as e:
            print(f"Skipping {split} due to error loading dataset: {e}")
            continue

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # 推理时不要打乱，以便与原始索引对应
            num_workers=args.num_workers,
            pin_memory=True
        )

        # 生成向量
        embeddings, redshifts = generate_embeddings(
            encoder,
            data_loader,
            device=args.device
        )

        # 4. --- Save Results ---
        output_dir = os.path.join(args.output_dir,args.ckpt)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{split}_photometry_embedding.npz")

        print(f"Embeddings shape: {embeddings.shape}")  # Should be (N, 6, 128)
        print(f"Redshifts shape: {redshifts.shape}")
        np.savez_compressed(
            output_filename,
            photometry_embeddings=embeddings,
            z=redshifts
        )
        print(f"Saved to {output_filename}")


if __name__ == '__main__':
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")

    parser = argparse.ArgumentParser(description="Generate embeddings using a pre-trained PhotometryEncoder.")

    parser.add_argument('--data_path', type=str, default=f"{ASTROCLIP_ROOT}/data/data_g3_z", help="Root directory of the train/test datasets.")
    parser.add_argument('--ckpt', type=str, default=f"photoencoder_00",
                        help="Path to the pre-trained photometry_encoder.pth checkpoint.")
    parser.add_argument('--output_dir', type=str, default=f"{ASTROCLIP_ROOT}/pretrained/embeddings",
                        help="Directory to save the output .npz files.")


    # Hardware / DataLoader
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 简单的路径检查
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

    main(args)
