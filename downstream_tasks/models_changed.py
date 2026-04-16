
import tempfile
from pathlib import Path

import lightning as L
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from numpy import ndarray
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FewShotRegressorModule(L.LightningModule):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dims: list[int],
        lr: float,
        dropout: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP(
            n_in=n_in,
            n_out=n_out,
            n_hidden=hidden_dims,
            act=[nn.ReLU()] * (len(hidden_dims) + 1),
            dropout=dropout,
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        outputs = self(inputs)
        if self.hparams.n_out == 1:
            outputs = outputs.squeeze(-1)
        loss = self.loss_fn(outputs, labels)
        return loss, outputs

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _ = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _ = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        inputs = batch[0]
        outputs = self(inputs)
        if self.hparams.n_out == 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



def few_shot(
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    max_epochs: int = 200,
    hidden_dims: list[int] = [64, 64],
    lr: float = 1e-3,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    patience: int = 15,
    min_delta: float = 1e-4,
    dropout: float = 0.1,
    seed: int = 42,
) -> ndarray:
    """Train a few-shot model with Lightning and early stopping."""
    seed_everything(seed, workers=True)

    y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    num_features = y_train_2d.shape[1]
    pin_memory = torch.cuda.is_available()

    if len(X_train) > 1 and val_fraction > 0:
        test_size = max(1, int(round(len(X_train) * val_fraction)))
        test_size = min(test_size, len(X_train) - 1)
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train,
            y_train_2d,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    else:
        X_train_split, y_train_split = X_train, y_train_2d
        X_val, y_val = None, None

    train_labels = y_train_split.ravel() if num_features == 1 else y_train_split
    train_dataset = TensorDataset(
        torch.tensor(X_train_split, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    val_loader = None
    callbacks = []
    checkpoint_dir = tempfile.TemporaryDirectory()
    if X_val is not None and y_val is not None:
        val_labels = y_val.ravel() if num_features == 1 else y_val
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.float32),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            dirpath=checkpoint_dir.name,
            filename="few-shot-best",
        )
        callbacks = [
            checkpoint_callback,
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
                min_delta=min_delta,
            ),
        ]
    else:
        checkpoint_callback = None

    module = FewShotRegressorModule(
        n_in=X_train.shape[1],
        n_out=num_features,
        hidden_dims=hidden_dims,
        lr=lr,
        dropout=dropout,
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = module
    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_model = FewShotRegressorModule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            n_in=X_train.shape[1],
            n_out=num_features,
            hidden_dims=hidden_dims,
            lr=lr,
            dropout=dropout,
        )

    predict_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    pred_batches = trainer.predict(best_model, dataloaders=predict_loader)
    preds = torch.cat(pred_batches, dim=0).cpu().numpy()

    checkpoint_dir.cleanup()
    return preds.ravel() if num_features == 1 else preds


def zero_shot(
    X_train: ndarray, y_train: ndarray, X_test: ndarray, n_neighbors: int = 64
) -> ndarray:
    """Train a zero-shot model using KNN"""
    neigh = KNeighborsRegressor(weights="distance", n_neighbors=64)
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    return preds


class MLP(nn.Sequential):
    """MLP model"""

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):
        if act is None:
            act = [
                nn.LeakyReLU(),
            ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_) - 2):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))
        layer.append(nn.Linear(n_[-2], n_[-1]))
        super(MLP, self).__init__(*layer)