import os
import sys
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..astrophoto.model import PhotometryEncoder
from ..modules import MLP, CrossAttentionHead
from .specformer import SpecFormer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dinov2.eval.setup import setup_and_build_model


class AstroClipModel(L.LightningModule):
    def __init__(
            self,
            image_encoder: nn.Module,
            spectrum_encoder: nn.Module,
            photometry_encoder: nn.Module,  # New argument
            temperature: float = 15.5,
            lr: float = 1e-4,
            weight_decay: float = 0.05,
            epochs: int = 100,
            eta_min: float = 5e-7,
            logit_scale: float = 15.5,
            learnable_logit_scale: bool = False,
            # Loss weights for each pair
            w_is: float = 1.0,  # image-spectrum
            w_ip: float = 1.0,  # image-photometry
            w_sp: float = 1.0,  # spectrum-photometry
    ):
        """
        The AstroCLIP model that takes an image and a spectrum and embeds them into a common space using CLIP loss.
        Note that you must provide the image and spectrum encoders to be used for the embedding.

        Args:
            image_encoder (nn.Module): The image encoder to be used for embedding.
            spectrum_encoder (nn.Module): The spectrum encoder to be used for embedding.
            temperature (float): The temperature parameter for the CLIP loss.
            lr (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            epochs (int): The number of epochs for training.
            eta_min (float): The minimum learning rate for the scheduler.
            logit_scale (float): The logit scale for the CLIP loss.
            learnable_logit_scale (bool): Whether the logit scale should be learnable.
        """
        super().__init__()
        self.save_hyperparameters()

        # Define the image and spectrum encoder
        self.image_encoder = image_encoder
        self.spectrum_encoder = spectrum_encoder
        self.photometry_encoder = photometry_encoder

        # Logit scale is fixed to 15.5 and is not a learnable parameter
        if not learnable_logit_scale:
            self.logit_scale = np.log(logit_scale)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))

        # Use CLIP loss
        self.criterion = CLIPLoss()

    def forward(
            self,
            input: torch.Tensor,
            input_type: str,
    ):
        if input_type == "image":
            return self.image_encoder(input)

        elif input_type == "spectrum":
            return self.spectrum_encoder(input)

        elif input_type == "photometry":
            return self.photometry_encoder(input)

        else:
            raise ValueError("Input type must be either 'image' or 'spectrum'")

    def _calculate_losses_for_scale(self, image_features, spectrum_features, photometry_features, scale, prefix):
        """
        Helper to compute all pairwise losses for a given scale and log them.
        Returns the total weighted loss for this scale.
        """
        loss_is = self.criterion(image_features, spectrum_features, scale)
        loss_ip = self.criterion(image_features, photometry_features, scale)
        loss_sp = self.criterion(spectrum_features, photometry_features, scale)

        total_loss = (
                self.hparams.w_is * loss_is +
                self.hparams.w_ip * loss_ip +
                self.hparams.w_sp * loss_sp
        )

        # Log individual and total losses with the given prefix
        self.log(f"{prefix}_loss_is", loss_is, sync_dist=True)
        self.log(f"{prefix}_loss_ip", loss_ip, sync_dist=True)
        self.log(f"{prefix}_loss_sp", loss_sp, sync_dist=True)
        self.log(f"{prefix}_total_loss", total_loss, sync_dist=True, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        im, sp, ph = batch["image"], batch["spectrum"], batch["params"]

        # Get features for all three modalities
        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)
        photometry_features = self.photometry_encoder(ph)

        # --- Calculate loss for backpropagation ---
        # Use the direct temperature value
        train_loss_withlogit = self._calculate_losses_for_scale(
            image_features, spectrum_features, photometry_features,
            scale=self.hparams.temperature,
            prefix="train_withlogit"
        )

        # --- Calculate loss for logging/monitoring ONLY ---
        # Use the log-transformed temperature value
        with torch.no_grad():
            train_loss_nologit = self._calculate_losses_for_scale(
                image_features, spectrum_features, photometry_features,
                scale=self.logit_scale,
                prefix="train_nologit"
            )

        # Log the scale values for reference
        self.log("scale_withlogit", self.hparams.temperature, sync_dist=True)
        self.log("scale_nologit", self.logit_scale, sync_dist=True)
        return train_loss_withlogit

    def validation_step(self, batch, batch_idx):
        im, sp, ph = batch["image"], batch["spectrum"], batch["params"]

        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)
        photometry_features = self.photometry_encoder(ph)

        # --- Calculate validation loss with the direct temperature value ---
        val_loss_withlogit = self._calculate_losses_for_scale(
            image_features, spectrum_features, photometry_features,
            scale=self.hparams.temperature,
            prefix="val_withlogit"
        )

        # --- Calculate validation loss with the log-transformed temperature value ---
        val_loss_nologit = self._calculate_losses_for_scale(
            image_features, spectrum_features, photometry_features,
            scale=self.logit_scale,
            prefix="val_nologit"
        )


class CLIPLoss(nn.Module):
    def get_logits(
            self,
            image_features: torch.FloatTensor,
            spectrum_features: torch.FloatTensor,
            logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        return logits_per_image, logits_per_image.T

    def forward(
            self,
            image_features: torch.FloatTensor,
            spectrum_features: torch.FloatTensor,
            logit_scale: float,
            output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the image and spectrum features
        logits_per_image, logits_per_spectrum = self.get_logits(
            image_features, spectrum_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_image.shape[0], device=image_features.device, dtype=torch.long
        )
        total_loss = (
                             F.cross_entropy(logits_per_image, labels)
                             + F.cross_entropy(logits_per_spectrum, labels)
                     ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss


class ImageHead(nn.Module):
    def __init__(
            self,
            config: str,
            model_weights: str,
            save_directory: str,
            embed_dim: int = 1024,
            n_head: int = 4,
            model_embed_dim: int = 1024,
            dropout: float = 0.1,
            freeze_backbone: bool = True,
    ):
        """
        Cross-attention image module that takes token outputs from the AstroDINO model and passes them through a
        cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_directory (str): Path to the directory containing the AstroDINO model.
            config (str): Path to the configuration file of the AstroDINO model.
            model_weights (str): Path to the weights of the AstroDINO model.
            embed_dim (int): Dimension of the AstroCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the AstroDINO embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the AstroDINO model.
        """
        super().__init__()

        # Define DINO config
        class config:
            output_dir = save_directory
            config_file = config
            pretrained_weights = model_weights
            opts = []

        # Define DINO model
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to null
        self.backbone, _ = setup_and_build_model(config())
        sys.stdout = sys.__stdout__  # Reset stdout

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # Set up MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.tensor, return_weights: bool = False):
        # Pass through the backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            x = self.backbone.patch_embed(x)
            for blk in self.backbone.blocks:
                x = blk(x)
            embedding = self.backbone.norm(x)

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x = x + self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()


class SpectrumHead(nn.Module):
    def __init__(
            self,
            model_path: str,
            embed_dim: int = 1024,
            n_head: int = 4,
            model_embed_dim: int = 768,
            dropout: float = 0.1,
            freeze_backbone: bool = True,
            load_pretrained_weights=True,
    ):
        """
        Cross-attention spectrum module that takes a spectrum and passes it through a pretrained SpecFormer model and
        then through a cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the AstroCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the SpecFormer embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the SpecFormer model.
        """
        super().__init__()
        # Load the model from the checkpoint
        checkpoint = torch.load(model_path)
        self.backbone = SpecFormer(**checkpoint["hyper_parameters"]['init_args'])
        if load_pretrained_weights:
            self.backbone.load_state_dict(checkpoint["state_dict"])

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # Set up MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(
            self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        # Embed the spectrum using the pretrained model
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["embedding"]

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x = x + self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()


class PhotometryHead(nn.Module):
    """
    A head for photometry data. It uses a pre-trained PhotometryEncoder as the backbone,
    followed by a projection head to map it to the shared embedding space.
    """

    def __init__(
            self,
            model_path: str,
            embed_dim: int = 1024,
            model_embed_dim: int = 768,
            dropout: float = 0.1,
            freeze_backbone: bool = True,
            load_pretrained_weights: bool = True,
    ):
        """
        Args:
            model_path (str): Path to the pre-trained PhotometryEncoder checkpoint (.pth).
            embed_dim (int): Dimension of the final shared AstroCLIP embedding (e.g., 1024).
            model_embed_dim (int): Dimension of the backbone encoder's embedding (e.g., 768).
            freeze_backbone (bool): Whether to freeze the backbone encoder's weights.
            load_pretrained_weights (bool): Whether to load pre-trained weights.
        """
        super().__init__()

        self.backbone = PhotometryEncoder(input_dim=19, embedding_dim=model_embed_dim)
        if load_pretrained_weights:
            self.backbone.load_state_dict(torch.load(model_path))
            print("Pre-trained photometry backbone weights loaded successfully.")

            # try:
            #     self.backbone.load_state_dict(torch.load(model_path))
            #     print("Pre-trained photometry backbone weights loaded successfully.")
            # except FileNotFoundError:
            #     print(f"Warning: Photometry checkpoint not found at {model_path}. Initializing from scratch.")
            # except Exception as e:
            #     print(f"Error loading photometry weights: {e}. Initializing from scratch.")

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # The "Projection Head" to map from model_embed_dim to the final embed_dim
        # This keeps the architecture consistent with ImageHead and SpectrumHead
        self.projection = nn.Linear(model_embed_dim, embed_dim)

        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)

        # Pass through the projection head
        x = self.projection(embedding)

        x = x + self.mlp(x)
        return x
