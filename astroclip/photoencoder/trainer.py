import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from astroclip import format_with_env
# 导入你之前定义的模块
from astroclip.photoencoder.data.datamodule import PhotometryDataModule
from astroclip.photoencoder.model.photoEncoder import MaskedPhotometryModel

ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")


def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="Pre-train Photometry Transformer")

    # Path args
    parser.add_argument('--data_path', type=str, default=f"{ASTROCLIP_ROOT}/data/data_g3_z_1k",
                        help='Path to dataset directory')

    # Training args
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4)

    # Model args
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--bias_scale', type=float, default=1.0, help='Initial scale for error bias')

    # Wandb args
    parser.add_argument('--wandb_project', type=str, default='astro-photometry-pretrain')
    parser.add_argument('--run_name', type=str, default="00")

    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    # 2. 设置随机种子 (保证复现性)
    pl.seed_everything(args.seed, workers=True)

    # 3. 准备 DataModule
    dm = PhotometryDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 4. 准备 Model
    model = MaskedPhotometryModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        bias_scale=args.bias_scale
    )

    # 5. 设置 WandB Logger
    wandb_logger = WandbLogger(
        entity=format_with_env("{WANDB_ENTITY_NAME}"),
        project=args.wandb_project,
        name=args.run_name,
        save_dir=f"{ASTROCLIP_ROOT}/outputs/photo_encoder",
        mode="offline"
    )

    # 6. 设置 Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ASTROCLIP_ROOT}/outputs/photo_encoder/{args.run_name}",
        filename='photo-transformer-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    # For stopping training early if validation loss doesn't improve
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 7. Trainer 初始化
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # 自动检测 GPU/MPS/CPU
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor,early_stopping_callback],
        deterministic=True  # 确保操作的确定性 (配合 seed_everything)
    )

    # 8. 开始训练
    print(f"Starting training with seed {args.seed}...")
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
