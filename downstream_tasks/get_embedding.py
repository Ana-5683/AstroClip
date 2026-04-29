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
        data: list,
        batch_size: int = 512,
) -> np.ndarray:
    """Get embeddings for data using a single model."""
    model_embeddings = []
    data_batch = []
    # data_batch_c=[]
    for item in tqdm(data):
        # 兼容 datasets 返回 torch.Tensor 或 numpy.ndarray 的两种情况。
        if not isinstance(item, torch.Tensor):
            item = torch.as_tensor(item, dtype=torch.float32)
        else:
            item = item.detach().to(dtype=torch.float32)

        # Add a batch dimension to the item tensor
        data_batch.append(item[None, ...])

        if len(data_batch) == batch_size:
            with torch.no_grad():
                # Create a batch tensor and move it to the GPU
                batch_tensor = torch.cat(data_batch).cuda()
                # The provided model function handles inference and returns a numpy array
                embeddings = model(batch_tensor)
                model_embeddings.append(embeddings)
            data_batch = []

    # Process the final batch if it's not empty
    if len(data_batch) > 0:
        with torch.no_grad():
            batch_tensor = torch.cat(data_batch).cuda()
            embeddings = model(batch_tensor)
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
    ckpt_name=f"{ckpt}"
    # Select and configure the requested model
    if model_name in ["astroclip_image", "astroclip_spectrum","astroclip_photo"]:
        astroclip_ckpt = os.path.join(pretrained_dir, f"{ckpt_name}.ckpt")
        astroclip = AstroClipModel.load_from_checkpoint(checkpoint_path=astroclip_ckpt)

        astroclip.eval()

        if model_name == "astroclip_image":
            data_key = "image"
            input_type = "image"
        elif model_name == "astroclip_spectrum":
            data_key = "spectrum"
            input_type = "spectrum"
        else:
            raise ValueError(f"Model '{model_name}' is not recognized.")

        model = lambda x: astroclip(x, input_type=input_type).cpu().numpy()
    elif model_name == "specformer":
        specformer_ckpt = os.path.join(pretrained_dir, f"{ckpt_name}.ckpt")
        checkpoint = torch.load(specformer_ckpt)
        specformer = SpecFormer(**checkpoint["hyper_parameters"])
        specformer.load_state_dict(checkpoint["state_dict"])
        specformer.cuda()

        data_key = "spectrum"
        model = lambda x: np.mean(specformer(x)["embedding"].cpu().numpy(), axis=1)
    elif model_name == "astrodino":
        astrodino_ckpt = os.path.join(pretrained_dir, f"{ckpt_name}.ckpt")
        astrodino = setup_astrodino(astrodino_output_dir, astrodino_ckpt)
        data_key = "image"
        model = lambda x: astrodino(x).cpu().numpy()
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")

    print(f"Model '{model_name}' has been set up successfully!")

    # --- Data Loading and Embedding Generation ---
    files = {"train": file_train, "test": file_test}
    for split, f in files.items():
        print(f"Processing '{split}' dataset...")
        dataset = load_from_disk(f)
        data_to_embed = dataset[data_key]

        # Generate embeddings for the current dataset split
        embeddings = get_single_embedding(model, data_to_embed, batch_size)

        # --- Save Embeddings ---
        npz_dict = {f"{model_name}_embeddings": embeddings}
        if 'z' in dataset.column_names:
            npz_dict['z'] = np.array(dataset['z'])
        if 'params' in dataset.column_names:
            npz_dict['params'] = np.array(dataset['params'])

        if model_name in ["astroclip_image", "astroclip_spectrum", "astroclip_photo"]:
            f_output_dir=os.path.join(output_dir,ckpt_name,model_name)
        else:
            f_output_dir=os.path.join(output_dir, ckpt_name)
        os.makedirs(f_output_dir, exist_ok=True)

        output_filename = os.path.join(f_output_dir,f"{split}_{model_name}_embedding.npz")
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
        choices=["astroclip_image", "astroclip_spectrum", "astrodino", "specformer"],
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
