# trainer.py

import argparse
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# Import our custom modules
from torch.utils.data import DataLoader

from astroclip import format_with_env
from astroclip.astrophoto.data import PhotometryTransform, PhotometryDataset
from astroclip.astrophoto.model import MFMPretrainer

torch.set_float32_matmul_precision('high')

ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
seed_everything(42, workers=True)


def main(args):
    # 1. --- Setup Data ---
    # --- W&B modification: Initialize W&B Run ---
    # The logger will automatically handle wandb.init()
    # You can customize project, name, etc.
    wandb_logger = WandbLogger(
        entity=format_with_env("{WANDB_ENTITY_NAME}"),
        project="astrophoto",
        name=args.run_name,
        save_dir=f"{ASTROCLIP_ROOT}/outputs/astrophoto",
        mode="offline",
    )

    transform = PhotometryTransform()

    # Create Datasets
    train_dataset = PhotometryDataset(data_path=args.data_path, split='train', transform=transform)
    val_dataset = PhotometryDataset(data_path=args.data_path, split='test', transform=transform)  # Using test as val

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. --- Setup Model ---
    model = MFMPretrainer(
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        masking_ratio=args.masking_ratio,
        learning_rate=args.learning_rate
    )

    # W&B: Watch the model for gradients and topology
    wandb_logger.watch(model, log='all', log_freq=100)

    # 3. --- Setup Training Callbacks and Logger ---
    # For saving the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{ASTROCLIP_ROOT}/outputs/astrophoto/{args.run_name}/ckpt',
        filename='photometry-pretrain-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    # For stopping training early if validation loss doesn't improve
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )

    # 4. --- Initialize and Start Trainer ---
    trainer = Trainer(
        accelerator='gpu',
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        precision=args.precision  # e.g., '16-mixed' for mixed precision
    )

    print("--- Starting MFM Pre-training ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("--- Training Finished ---")

    # 5. --- Save the Final Encoder ---
    # After training, we extract the trained encoder from the best checkpoint
    # and save it for Stage 2.
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")

    # Load the entire LightningModule from the checkpoint
    final_lightning_model = MFMPretrainer.load_from_checkpoint(best_model_path)

    # Extract the encoder's state dictionary
    encoder_state_dict = final_lightning_model.encoder.state_dict()

    # Save the encoder weights
    save_path = f"{ASTROCLIP_ROOT}/outputs/astrophoto/{args.run_name}/pretrained_photometry_encoder.ckpt"
    torch.save(encoder_state_dict, save_path)
    print(f"Successfully saved the pre-trained encoder weights to '{save_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Lightning MFM Pre-trainer for Photometry Data")

    # Data arguments
    parser.add_argument('--data_path', type=str, default=f"{ASTROCLIP_ROOT}/data/data_g3_z",
                        help="Root directory of the train/test datasets.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")

    # Model arguments
    parser.add_argument('--input_dim', type=int, default=19, help="Input dimension of the photometry features.")
    parser.add_argument('--embedding_dim', type=int, default=256, help="Output dimension of the encoder.")
    parser.add_argument('--masking_ratio', type=float, default=0.4, help="Ratio of features to mask.")

    # Training arguments
    parser.add_argument('--devices', type=str, default='-1', help="Number of devices to use for training.")
    parser.add_argument('--max_epochs', type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping.")
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help="Training precision (e.g., '32-true', '16-mixed').")

    # wandb setup
    parser.add_argument('--run_name', type=str, default="00", help="Name for the W&B run.")

    args = parser.parse_args()
    main(args)
