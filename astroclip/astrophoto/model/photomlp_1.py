import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import lightning as L

# ================================================================= #
#  Part 1: The Photometry Encoder Model (MLP)                     #
# ================================================================= #

class PhotometryEncoder(nn.Module):
    """
    MLP-based encoder for tabular photometry data.
    Takes a 19-dimensional feature vector and projects it into a
    high-dimensional embedding space.
    """

    def __init__(self, input_dim: int = 19, embedding_dim: int = 768,dropout:float=0.1):
        """
        Initializes the PhotometryEncoder.

        Args:
            input_dim (int): The number of input features (e.g., 19).
            embedding_dim (int): The dimension of the output embedding.
                                 We set this to 768 to align with the
                                 spectrum encoder's output dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.model = nn.Sequential(
            # Input LayerNorm for stabilizing training
            nn.LayerNorm(self.input_dim),

            # Hidden Layer 1
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            # nn.Dropout(dropout),

            # Hidden Layer 2
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(dropout),

            # Output Layer to produce the embedding
            # No activation function here, as we want a raw embedding vector.
            nn.Linear(512, self.embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, embedding_dim).
        """
        return self.model(x)


# ================================================================= #
#  Part 2: The Masked Feature Modeling (MFM) Pre-training Wrapper   #
# ================================================================= #

class MFMPretrainer(L.LightningModule):
    """
    A PyTorch Lightning Module for self-supervised pre-training of a
    photometry encoder using the Masked Feature Modeling (MFM) paradigm.
    """

    def __init__(self,
                 input_dim: int = 19,
                 embedding_dim: int = 768,
                 dropout: float = 0.1,
                 masking_ratio: float = 0.2,
                 learning_rate: float = 1e-4):
        """
        Initializes the MFM Pre-trainer Lightning Module.

        Args:
            input_dim (int): The number of input features.
            embedding_dim (int): The dimension of the encoder's output embedding.
            masking_ratio (float): The fraction of features to randomly mask.
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()
        # Save hyperparameters to self.hparams. This is a good Lightning practice.
        self.save_hyperparameters()

        # Instantiate the encoder and prediction head within the module
        self.encoder = PhotometryEncoder(input_dim, embedding_dim,dropout)
        self.prediction_head = nn.Linear(embedding_dim, input_dim)

    def _forward_and_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the core forward pass and loss calculation.
        """
        # 1. Create a random mask
        mask = torch.rand(batch.shape, device=self.device) < self.hparams.masking_ratio

        # Ensure at least one feature is masked to avoid NaN loss
        if not mask.any():
            # If no features were masked, randomly pick one to mask
            rand_idx = torch.randint(0, batch.shape[1], (1,)).item()
            mask[0, rand_idx] = True

        # 2. Create the corrupted input
        corrupted_x = batch.clone()
        corrupted_x[mask] = 0

        # 3. Get the embedding from the corrupted input
        embedding = self.encoder(corrupted_x)

        # 4. Predict the original features
        predicted_features = self.prediction_head(embedding)

        # 5. Calculate loss ONLY on the masked features
        loss = F.mse_loss(predicted_features[mask], batch[mask])
        return loss

    def training_step(self, batch, batch_idx):
        # `batch` is expected to be a tensor directly from the DataLoader
        loss = self._forward_and_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_and_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    # def test_step(self, batch, batch_idx):
    #     # We can reuse the validation logic for testing
    #     loss = self._forward_and_loss(batch)
    #     self.log('test_loss', loss, on_epoch=True)
    #     return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the training process.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

