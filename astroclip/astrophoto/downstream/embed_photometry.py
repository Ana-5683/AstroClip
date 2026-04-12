# embed_photometry.py

import argparse
import os

import datasets
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from astroclip import format_with_env
from astroclip.astrophoto.data import PhotometryTransform
from astroclip.astrophoto.model import PhotometryEncoder


# in dataset.py

class PhotometryDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train', transform=None):
        if split == 'train':
            data_path = os.path.join(data_path, 'train_dataset')
        elif split == 'test':
            data_path = os.path.join(data_path, 'test_dataset')

        self.dataset = datasets.load_from_disk(data_path)
        self.transform = transform

        # Set format for both columns we need
        self.dataset.set_format(type="torch", columns=['params', 'z'])
        print(f"Loaded {len(self.dataset)} samples from {data_path}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        params_tensor = sample['params']
        z_tensor = sample['z']

        # Apply the transform ONLY to the params tensor
        if self.transform:
            params_tensor = self.transform(params_tensor)

        # ALWAYS return both the (potentially transformed) params and the z value
        return params_tensor, z_tensor


def generate_embeddings(
        encoder: PhotometryEncoder,
        data_loader: DataLoader,
        device: str = "cuda"
) -> np.ndarray:
    """
    Generates embeddings for a given dataset using the provided encoder.
    """
    encoder.to(device)
    encoder.eval()  # Set the model to evaluation mode

    all_embeddings = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Generating Embeddings"):
            # The DataLoader already applies the PhotometryTransform,
            # so batch_data is the final 19-dim preprocessed tensor.
            batch_data = batch_data.to(device)

            # Get embeddings from the encoder
            # Note: We are using the preprocessed data to get embeddings,
            # NOT passing it through the MFM pretrainer wrapper.
            embeddings = encoder(batch_data)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings)


def main(args):
    # 1. --- Setup Data Transformation ---
    # The mean and std are hardcoded in PhotometryTransform as per your design.
    # We just need to instantiate it.
    transform = PhotometryTransform()

    # 2. --- Setup Model ---
    print(f"Loading pre-trained PhotometryEncoder from: {args.ckpt_path}")
    encoder = PhotometryEncoder(
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim
    )

    # Load the saved state dictionary
    try:
        state_dict = torch.load(args.ckpt_path)
        encoder.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.ckpt_path}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 3. --- Process Train and Test/Validation splits ---
    for split in ['train', 'test']:
        print(f"\n--- Processing '{split}' dataset ---")
        # Create dataset and dataloader

        # Initialize dataset WITH the transform.
        dataset = PhotometryDataset(data_path=args.data_path, split=split, transform=transform)

        # Let's create a temporary dataloader for the raw data for batching
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Important to keep order for z
            num_workers=args.num_workers
        )

        encoder.to(args.device)
        encoder.eval()

        all_embeddings = []
        all_redshifts = []

        with torch.no_grad():
            for transformed_params_batch, z_batch in tqdm(data_loader, desc=f"Embedding {split} split"):
                transformed_params_batch = transformed_params_batch.to(args.device)

                embeddings = encoder(transformed_params_batch)

                all_embeddings.append(embeddings.cpu().numpy())
                all_redshifts.append(z_batch.numpy())

        final_embeddings = np.concatenate(all_embeddings)
        final_redshifts = np.concatenate(all_redshifts)

        # 4. --- Save Embeddings and Redshifts ---
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.join(output_dir, f"{split}_photometry_embedding.npz")

        npz_dict = {
            "photometry_embeddings": final_embeddings,
            "z": final_redshifts
        }

        np.savez(output_filename, **npz_dict)
        print(f"Embeddings for '{split}' split saved to {output_filename}")


if __name__ == '__main__':
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")

    parser = argparse.ArgumentParser(description="Generate embeddings using a pre-trained PhotometryEncoder.")

    parser.add_argument('--data_path', type=str, default=f"{ASTROCLIP_ROOT}/data/data_g3_z", help="Root directory of the train/test datasets.")
    parser.add_argument('--ckpt_path', type=str, default=f"{ASTROCLIP_ROOT}/outputs/astrophoto/pretrained_photometry_encoder.ckpt",
                        help="Path to the pre-trained photometry_encoder.pth checkpoint.")
    parser.add_argument('--output_dir', type=str, default=f"{ASTROCLIP_ROOT}/pretrained/embeddings/astrophoto",
                        help="Directory to save the output .npz files.")

    # Model parameters must match the pre-trained encoder
    parser.add_argument('--input_dim', type=int, default=19, help="Input dimension after preprocessing.")
    parser.add_argument('--embedding_dim', type=int, default=256, help="Output dimension of the encoder.")

    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for generating embeddings.")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for DataLoader.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference.")

    args = parser.parse_args()
    main(args)