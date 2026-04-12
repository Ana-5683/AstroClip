import os
from argparse import ArgumentParser
from typing import Callable

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from astroclip.astrodino.utils import setup_astrodino
from astroclip.env import format_with_env
from astroclip.models import AstroClipModel, SpecFormer


def get_single_embedding(
        model: Callable,
        data_image: list,
        data_photo: list,
        batch_size: int = 512,
) -> np.ndarray:
    """Get embeddings for data using a single model."""
    model_embeddings = []
    data_batch_i = []
    data_batch_p = []

    for item_i, item_p in tqdm(zip(data_image, data_photo)):
        # Add a batch dimension to the item tensor
        # data_batch_i.append(torch.tensor(item_i, dtype=torch.float32)[None, ...])
        # data_batch_p.append(torch.tensor(item_p, dtype=torch.float32)[None, ...])
        data_batch_i.append(item_i.clone().detach()[None, ...])
        data_batch_p.append(item_p.clone().detach()[None, ...])

        if len(data_batch_i) == batch_size and len(data_batch_p) == batch_size:
            with torch.no_grad():
                # Create a batch tensor and move it to the GPU
                batch_tensor_i = torch.cat(data_batch_i).cuda()
                batch_tensor_p = torch.cat(data_batch_p).cuda()
                # The provided model function handles inference and returns a numpy array
                embeddings = model(batch_tensor_i, batch_tensor_p)
                model_embeddings.append(embeddings)
            data_batch_i = []
            data_batch_p = []

    # Process the final batch if it's not empty
    if len(data_batch_i) > 0 and len(data_batch_p) > 0:
        with torch.no_grad():
            batch_tensor_i = torch.cat(data_batch_i).cuda()
            batch_tensor_p = torch.cat(data_batch_p).cuda()
            embeddings = model(batch_tensor_i, batch_tensor_p)
            model_embeddings.append(embeddings)

    # Concatenate embeddings from all batches into a single numpy array
    return np.concatenate(model_embeddings)


def embed_provabgs(
        file_train: str,
        file_test: str,
        pretrained_dir: str,
        output_dir: str,
        model_name: str,
        ckpt: str,
        batch_size: int = 512,
):
    """
    Generates and saves embeddings for a specified model on training and test datasets.
    """
    # --- Model Setup ---
    astrodino_output_dir = os.path.join(pretrained_dir, "astrodino_output_dir")

    # Select and configure the requested model

    astroclip_ckpt = os.path.join(pretrained_dir, f"{ckpt}.ckpt")
    astroclip = AstroClipModel.load_from_checkpoint(checkpoint_path=astroclip_ckpt)

    astroclip.eval()
    # astroclip.cuda()

    data_key1 = "image"
    data_key2 = "params"
    input_type = "ip"
    model = lambda x1, x2: astroclip(x1, x2, input_type=input_type).cpu().numpy()
    print(f"Model '{model_name}' has been set up successfully!")

    # --- Data Loading and Embedding Generation ---
    files = {"train": file_train, "test": file_test}
    for split, f in files.items():
        print(f"Processing '{split}' dataset...")
        dataset = load_from_disk(f)
        data_to_embed1 = dataset[data_key1]
        data_to_embed2 = dataset[data_key2]

        # print(data_to_embed1[0].shape)

        # Generate embeddings for the current dataset split
        embeddings = get_single_embedding(model, data_to_embed1, data_to_embed2, batch_size)

        # --- Save Embeddings ---
        npz_dict = {f"{model_name}_embeddings": embeddings}
        if 'z' in dataset.column_names:
            npz_dict['z'] = np.array(dataset['z'])
        if 'params' in dataset.column_names:
            npz_dict['params'] = np.array(dataset['params'])

        f_output_dir=os.path.join(output_dir,ckpt,model_name)
        os.makedirs(f_output_dir, exist_ok=True)
        output_filename = os.path.join(f_output_dir, f"{split}_{model_name}_embedding.npz")

        np.savez(output_filename, **npz_dict)
        print(f"Embeddings for the '{split}' split have been saved to {output_filename}")


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "--dset",
        type=str,
        default=f"data_g3_z",
        help="Path to the training dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=f"c48_pe",
        help="dataset 后缀",
    )

    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=f"{ASTROCLIP_ROOT}/pretrained",
        help="Directory containing pretrained model checkpoints.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["astroclip_ip"],
        help="The model to use for generating embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size for processing.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{ASTROCLIP_ROOT}/pretrained/embeddings",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # 输出数据集名
    train_dir=os.path.join(ASTROCLIP_ROOT,"data",args.dset,f"train_dataset_preprocessed_{args.name}")
    test_dir=os.path.join(ASTROCLIP_ROOT,"data",args.dset,f"test_dataset_preprocessed_{args.name}")
    print(f"Using dataset: {args.dset}")
    print(f"Using train dataset: {os.path.basename(train_dir)}")
    print(f"Using test dataset: {os.path.basename(test_dir)}")

    embed_provabgs(
        train_dir,
        test_dir,
        args.pretrained_dir,
        args.output_dir,
        args.model,
        args.ckpt,
        args.batch_size
    )