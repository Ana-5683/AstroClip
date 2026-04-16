# calculate_stats.py

import argparse
import os
import torch
import datasets
from tqdm import tqdm

from astroclip import format_with_env


def calculate_photometry_stats(data_path, batch_size=1000):
    """
    Calculates the mean and std of SDSS photometry data for the (5, 2) shape.

    Structure: (Sequence=5, Features=2)
    Feature 0: Corrected Magnitude (psfMag - extinction)
    Feature 1: Log Magnitude Error (ln(psfMagErr))
    """

    # 1. Path Setup
    train_path = os.path.join(data_path, 'train_dataset')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at: {train_path}")

    print(f"Loading dataset from {train_path}...")
    # Load dataset (using memory mapping by default, which is efficient)
    dataset = datasets.load_from_disk(train_path)
    dataset.set_format(type="torch", columns=['params'])

    print(f"Total samples: {len(dataset)}")

    # 2. Indices Setup
    mag_indices = slice(0, 5)
    ext_indices = slice(5, 10)
    err_indices = slice(10, 15)
    epsilon = 1e-6

    # 3. Accumulators for Online Mean/Std Calculation (Welford's algorithm or simple accumulation)
    # Since dataset size is manageable for RAM in most astronomical cases (<10GB), 
    # we can try to load chunks or just accumulate sum and sum_squares.

    # To be safe with memory, we iterate in batches.
    n_samples = 0
    sum_x = torch.zeros(5, 2, dtype=torch.float64)  # Use float64 for precision
    sum_x2 = torch.zeros(5, 2, dtype=torch.float64)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

    print("Processing data to compute statistics...")

    for batch in tqdm(dataloader):
        # Shape: (B, 15)
        raw_data = batch['params']
        B = raw_data.shape[0]

        # --- Preprocessing Logic (Must match photo_augmentations.py) ---

        # A. Extract components
        raw_mags = raw_data[:, mag_indices]
        extinctions = raw_data[:, ext_indices]
        raw_errs = raw_data[:, err_indices]

        # B. Physics Correction: Mag - Extinction
        corrected_mags = raw_mags - extinctions

        # C. Log Transform Error
        # Note: We use raw_errs + epsilon inside log
        log_errs = torch.log(raw_errs + epsilon)

        # D. Stack to (B, 5, 2)
        # dim=2 stack: [mag, err]
        processed_batch = torch.stack([corrected_mags, log_errs], dim=2)

        # --- Accumulate ---
        # Sum across batch dimension
        sum_x += torch.sum(processed_batch, dim=0).double()
        sum_x2 += torch.sum(processed_batch ** 2, dim=0).double()
        n_samples += B

    # 4. Final Calculation
    mean = sum_x / n_samples

    # 正确计算样本标准差
    var = (sum_x2 / n_samples) - (mean ** 2)
    var = var * (n_samples / (n_samples - 1))  # 关键修正：从总体转为样本方差
    var = torch.clamp(var, min=1e-12)  # 更小的下限更安全
    std = torch.sqrt(var)

    # 5. Output
    print("\n" + "=" * 40)
    print("COMPLETED! COPY THE FOLLOWING TENSORS:")
    print("=" * 40)

    # Format for easy copy-pasting
    print("\n# Copy this into photo_augmentations.py -> __init__\n")

    print(f"self.mean = torch.tensor({mean.float().tolist()})")
    print(f"self.std = torch.tensor({std.float().tolist()})")

    print("\n" + "=" * 40)
    print("Interpretation:")
    print("Shape (5, 2) corresponds to:")
    print("Rows: [u, g, r, i, z]")
    print("Cols: [Corrected_Mag, Log_Error]")


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")

    parser = argparse.ArgumentParser(description="Calculate Mean and Std for Photometry Normalization")
    parser.add_argument('--dset_dir', type=str, default="data_g3_z",
                        help="Root directory containing 'train_dataset' folder")
    parser.add_argument('--batch_size', type=int, default="64")
    args = parser.parse_args()

    print(f"Using data path: {args.dset_dir}")
    data_path = os.path.join(ASTROCLIP_ROOT, 'data', args.dset_dir)
    calculate_photometry_stats(data_path, batch_size=args.batch_size)
