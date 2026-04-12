# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl


class AstroPhotometryTransformer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 embed_dim=128,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1,
                 bias_scale=1.0):
        """
        Transformer Encoder specifically designed for SDSS Photometry (SED) data.

        Args:
            input_dim (int): Input features per band (Default: 2 -> [Mag, LogErr]).
            embed_dim (int): Latent dimension size (d_model).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer Encoder layers.
            bias_scale (float): Initial scale factor for the error-based attention bias.
        """
        super().__init__()

        self.num_heads=num_heads
        self.embed_dim = embed_dim

        # 1. Feature Projection (Linear Embedding)
        # Projects (Mag, Err) -> Latent Space
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 2. Special Tokens
        # [CLS] token: To aggregate global SED features
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # [MASK] token: To replace masked bands during pre-training
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. Positional Embeddings
        # Learnable position for 6 tokens: [CLS, u, g, r, i, z]
        self.pos_embed = nn.Parameter(torch.randn(1, 6, embed_dim))

        # 4. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True  # Pre-Norm is generally more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Error Bias Scale
        # A learnable scalar to control how strongly the Error affects Attention
        # Bias = - abs(scale) * Error
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale))

        # 6. Pre-training Head (Reconstruction)
        # Predicts the single masked magnitude value
        self.head_pred = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # Initialization suitable for Transformers
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.head_pred.weight)

    def build_error_bias(self, log_errs):
        """
        Constructs the Attention Bias Matrix based on Log Errors.

        Args:
            log_errs: (Batch, 5) Standardized Log-Errors.
                      High value = High Error = Low Attention Weight.

        Returns:
            attn_bias: (Batch*Num_Heads, 6, 6) or (Batch, 6, 6) depending on usage.
        """
        batch_size = log_errs.shape[0]

        # 1. Expand Errors to include CLS token
        # CLS token is internally generated, so we assume its error is 0 (or minimum)
        # We use a very small value or 0 depending on normalization.
        # Assuming standardized input, let's use the min of the batch or simply -2 (low error).
        # Here we use 0 assuming standardized data is centered.
        cls_errs = torch.zeros(batch_size, 1, device=log_errs.device)

        # Shape: (Batch, 6) -> [CLS_Err, u_Err, ..., z_Err]
        full_errs = torch.cat([cls_errs, log_errs], dim=1)

        # 2. Create Bias Matrix (Source-based)
        # If Source j has high error, Target i should not attend to j.
        # So Bias_{i,j} depends on Error_{j}.

        # Shape: (Batch, 1, 6) -> Broadcast to (Batch, 6, 6)
        # Rows (Targets) are identical, Columns (Sources) vary.
        bias_matrix = full_errs.unsqueeze(1).repeat(1, 6, 1)

        # 3. Apply Scaling
        # We only want to penalize high errors.
        # Logic: Attention_Score + (- lambda * Error)
        # We use softplus to ensure scale is positive
        scale = F.softplus(self.bias_scale)
        attn_bias = - scale * bias_matrix

        return attn_bias

    def forward(self, x, mask_indices=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 5, 2).
                              x[:, :, 0] is Magnitude, x[:, :, 1] is LogError.
            mask_indices (torch.Tensor, optional): Shape (Batch,). Indices (0-4) to mask.
                              If None, no masking is performed (Inference mode).

        Returns:
            dict: {
                "features": (Batch, 6, 128) - The full sequence of tokens.
                "pred_mag": (Batch, 1) - Predicted magnitude for the masked band (if masked).
            }
        """
        B, L, C = x.shape  # (Batch, 5, 2)

        # --- 1. Separate Mag and Err ---
        mags = x[:, :, 0:1]  # (B, 5, 1)
        log_errs = x[:, :, 1]  # (B, 5) -> We keep it 1D for bias construction

        # --- 2. Input Projection ---
        # (B, 5, 2) -> (B, 5, 128)
        x_embed = self.input_proj(x)

        # --- 3. Apply Masking (Pre-training Strategy) ---
        if mask_indices is not None:
            # mask_indices contains values 0 to 4.
            # We construct a batch indexer
            batch_idx = torch.arange(B, device=x.device)

            # Replace the embedding at the mask index with the learnable [MASK] token
            # Note: We must clone to avoid in-place modification error in gradient
            x_embed = x_embed.clone()
            x_embed[batch_idx, mask_indices] = self.mask_token.squeeze(0).expand(B, -1)

            # NOTE regarding Attention Bias during Masking:
            # Even if a token is masked, its Error value in `log_errs` is still the REAL error.
            # Technically, we should set the error of the MASK token to 0 (neutral),
            # because the "container" itself is not noisy, it's just empty.
            # Let's zero-out the error for the masked position to allow others to attend to MASK
            # (though usually others attend to unmasked, and MASK attends to others).
            # To be safe, let's keep the bias as is, or zero it.
            # Strategy: Zero it out so the MASK token is not penalized.
            log_errs_for_bias = log_errs.clone()
            log_errs_for_bias[batch_idx, mask_indices] = 0.0
        else:
            log_errs_for_bias = log_errs

        # --- 4. Append CLS Token ---
        # Expand CLS to (B, 1, 128)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Concat: [CLS, u, g, r, i, z] -> (B, 6, 128)
        x_seq = torch.cat((cls_tokens, x_embed), dim=1)

        # --- 5. Add Position Embedding ---
        x_seq = x_seq + self.pos_embed

        # --- 6. Compute Attention Bias ---
        # Returns (B, 6, 6)
        attn_bias = self.build_error_bias(log_errs_for_bias)

        # 使用 repeat_interleave 扩展为 (B*H, 6, 6)
        # 这在 batch_first=False 模式下是完全合法的
        attn_bias = attn_bias.repeat_interleave(self.num_heads, dim=0)

        # --- 【关键修改 2】: 手动转置以适配 batch_first=False ---
        # (B, L, E) -> (L, B, E)
        x_seq = x_seq.transpose(0, 1)

        # 此时 attn_bias 形状为 (256, 4, 6, 6)，完美匹配 input (256, 4, 6, 6)
        # 【关键修改 END】

        # PyTorch Transformer expects mask shape:
        # (Batch * num_heads, Target, Source) or (Batch, Target, Source)
        # We need to replicate for heads is handled automatically by PyTorch >= 1.9
        # if mask is (B, T, S). Let's Check PyTorch version compatibility in mind.
        # Ideally, we pass (B, 6, 6).

        # --- 7. Transformer Encoder Forward ---
        # Note: src_mask is added to attention scores.
        # Check if PyTorch version supports batch-wise float mask.
        # If you encounter errors, you might need repeat_interleave for heads.
        features = self.transformer(x_seq, mask=attn_bias)

        # --- 【关键修改 3】: 转置回来 ---
        # (L, B, E) -> (B, L, E)
        features = features.transpose(0, 1)

        # --- 8. Prediction Head (for Masked Element) ---
        pred_mag = None
        if mask_indices is not None:
            # We need to extract the features at the masked positions
            # Indices in features are shifted by +1 because of [CLS]
            target_indices = mask_indices + 1

            batch_idx = torch.arange(B, device=x.device)
            masked_features = features[batch_idx, target_indices]  # (B, 128)

            pred_mag = self.head_pred(masked_features)  # (B, 1)

        return {
            "features": features,  # (B, 6, 128) -> Output for Multi-modal Alignment
            "pred_mag": pred_mag  # (B, 1) -> Output for Pre-training Loss
        }

class MaskedPhotometryModel(pl.LightningModule):
    """
    Lightning 训练包装器：定义 Training Step, Optimizer 等
    """

    def __init__(self,
                 d_model=128,
                 n_heads=4,
                 n_layers=3,
                 lr=1e-3,
                 weight_decay=1e-4,
                 bias_scale=1.0):
        super().__init__()
        self.save_hyperparameters()  # 自动记录超参数到 wandb

        # 实例化核心模型
        self.model = AstroPhotometryTransformer(
            input_dim=2,
            embed_dim=d_model,
            num_heads=n_heads,
            num_layers=n_layers,
            dropout=0.1,
            bias_scale=bias_scale
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, mask_indices=None):
        return self.model(x, mask_indices=mask_indices)

    def training_step(self, batch, batch_idx):
        # batch shape: (B, 5, 2)
        # x[:,:,0] is Normalized Mag, x[:,:,1] is Normalized LogErr
        x = batch
        batch_size = x.shape[0]

        # 1. 生成随机 Mask 索引 (0-4)
        mask_indices = torch.randint(0, 5, (batch_size,), device=self.device)

        # 2. 前向传播
        output = self.model(x, mask_indices=mask_indices) # (B,1)

        pred_mag=output["pred_mag"]

        # 3. 获取 Ground Truth
        # 我们要预测的是被 Mask 掉的那个波段的 Normalized Corrected Mag
        # x 的维度是 (B, 5, 2)，取第 0 个特征 (Mag)
        # 利用 gather 或者高级索引提取目标值
        target_mag = x[torch.arange(batch_size, device=self.device), mask_indices, 0].unsqueeze(1)

        # 4. 计算 Loss (MSE)
        loss = F.mse_loss(pred_mag, target_mag)

        # 5. Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证步逻辑：同样做 Masked Prediction，看看重建误差
        x = batch
        batch_size = x.shape[0]

        # 为了验证的确定性，我们可以固定 mask 掉中间波段 (e.g., r-band index 2)
        # 或者依然随机
        mask_indices = torch.randint(0, 5, (batch_size,), device=self.device)

        output = self.model(x, mask_indices=mask_indices) # (B,1)

        pred_mag = output["pred_mag"]

        target_mag = x[torch.arange(batch_size, device=self.device), mask_indices, 0].unsqueeze(1)

        val_loss = F.mse_loss(pred_mag, target_mag)
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # 使用 Cosine Annealing 调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }
