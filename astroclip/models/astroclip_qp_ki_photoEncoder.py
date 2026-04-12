import os
import sys
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..photoencoder.model.photoEncoder import MaskedPhotometryModel
from ..scheduler import CosineAnnealingWithWarmupLR
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
            temperature: float = 15.5,
            lr: float = 1e-4,
            weight_decay: float = 0.05,
            epochs: int = 100,
            eta_min: float = 5e-7,
            logit_scale: float = 15.5,
            learnable_logit_scale: bool = False,
            layer_decay: float = 0.7,  # LLRD 衰减系数
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

        # Logit scale is fixed to 15.5 and is not a learnable parameter
        if not learnable_logit_scale:
            self.logit_scale = torch.tensor(np.log(logit_scale))
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))

        # Use CLIP loss
        self.criterion = CLIPLoss()

    def forward(
            self,
            input: torch.Tensor,
            input_p: torch.Tensor,
            input_type: str,
    ):
        if input_type == "ip":
            return self.image_encoder(input, input_p)

        elif input_type == "spectrum":
            return self.spectrum_encoder(input)

        else:
            raise ValueError("Input type must be either 'image' or 'spectrum'")

    def training_step(self, batch, batch_idx):
        im, sp, photo = batch["image"], batch["spectrum"], batch["params"]

        # 1. 限制 Logit Scale 范围 (防止梯度爆炸)
        # Max: ln(100) ≈ 4.605
        if isinstance(self.logit_scale, nn.Parameter):
            with torch.no_grad():
                self.logit_scale.clamp_(max=4.6052)

        # 2. 获取实际 Scale (Exp)
        # 注意：这里我们取 exp，让 Scale 回到 15.5~100 的线性空间
        actual_scale = self.logit_scale.exp()

        # 1. 获取本地特征 (不进行归一化，Raw Features)
        local_image_features = self.image_encoder(im, photo)
        local_spectrum_features = self.spectrum_encoder(sp)

        # 2. 跨显卡聚合特征
        if self.trainer.world_size > 1:
            # 此时聚合的是原始特征，数值范围可能不是 [-1, 1]
            # 但这没关系，后续的 Loss 会处理
            all_image_features = self.all_gather(local_image_features, sync_grads=True)
            all_spectrum_features = self.all_gather(local_spectrum_features, sync_grads=True)

            # Reshape [4, 256, D] -> [1024, D]
            all_image_features = all_image_features.view(-1, all_image_features.shape[-1])
            all_spectrum_features = all_spectrum_features.view(-1, all_spectrum_features.shape[-1])
        else:
            all_image_features = local_image_features
            all_spectrum_features = local_spectrum_features

        # print("=====================================================")
        # print("=====================================================")
        # print("=====================================================")
        # print("all_image_features.shape: ", all_image_features.shape)
        # print("all_spectrum_features.shape: ", all_spectrum_features.shape)
        # print("=====================================================")
        # print("=====================================================")
        # print("=====================================================")

        # Calculate the CLIP loss
        train_loss_withlogit = self.criterion(
            all_image_features, all_spectrum_features, actual_scale
        )
        # train_loss_nologit = self.criterion(
        #     all_image_features, all_spectrum_features, actual_scale
        # )

        # Log the losses
        self.log("train_loss_withlogit", train_loss_withlogit, prog_bar=True)
        # self.log("train_loss_nologit", train_loss_nologit, sync_dist=True, prog_bar=True)
        self.log("scale", actual_scale)

        # Return the loss
        return train_loss_withlogit

    def validation_step(self, batch, batch_idx):
        im, sp, photo = batch["image"], batch["spectrum"], batch["params"]

        # 1. 获取本地特征 (不进行归一化，Raw Features)
        local_image_features = self.image_encoder(im, photo)
        local_spectrum_features = self.spectrum_encoder(sp)

        # 2. 跨显卡聚合特征 (为了保持与训练时一致的负样本数量)
        if self.trainer.world_size > 1:
            # 注意：这里不需要 sync_grads=True，因为验证阶段不更新梯度
            all_image_features = self.all_gather(local_image_features)
            all_spectrum_features = self.all_gather(local_spectrum_features)

            # Reshape [4, 256, D] -> [1024, D]
            all_image_features = all_image_features.view(-1, all_image_features.shape[-1])
            all_spectrum_features = all_spectrum_features.view(-1, all_spectrum_features.shape[-1])
        else:
            all_image_features = local_image_features
            all_spectrum_features = local_spectrum_features

        # 计算 Loss (仅作参考)
        actual_scale = self.logit_scale.exp()

        # Calculate the CLIP loss
        val_loss_withlogit = self.criterion(
            all_image_features, all_spectrum_features, actual_scale
        )
        # val_loss_nologit = self.criterion(
        #     all_image_features, all_spectrum_features, actual_scale
        # )

        # Log the losses
        self.log("val_loss_withlogit", val_loss_withlogit, prog_bar=True)
        # self.log("val_loss_nologit", val_loss_nologit, sync_dist=True)

        # recall@K
        r1, r5 = self.recall_k(all_img_feats=all_image_features, all_spec_feats=all_spectrum_features)
        # 4. 记录日志 (这才是你的“体检报告”)
        self.log("val/R1", r1)
        self.log("val/R5", r5)

    def recall_k(self, all_img_feats, all_spec_feats):
        # 2. 计算相似度矩阵 (不需要乘 Scale，因为我们只看排序)
        # 归一化
        all_img_feats = F.normalize(all_img_feats, dim=-1,eps=1e-3)
        all_spec_feats = F.normalize(all_spec_feats, dim=-1,eps=1e-3)

        # [Batch_Size, Batch_Size]
        logits = all_img_feats @ all_spec_feats.T

        # 3. 计算 Recall@K (R@1 和 R@5)
        batch_size = logits.shape[0]
        targets = torch.arange(batch_size, device=logits.device)  # 对角线是正确答案

        # image-to-spectrum retrieval
        # 获取每张图预测的前5个光谱的索引
        _, topk_indices = logits.topk(k=5, dim=1)

        # 计算 R@1: 预测的第一名 == 真实标签
        r1 = (topk_indices[:, 0] == targets).float().mean()

        # 计算 R@5: 真实标签 是否出现在 前5名中
        # targets.view(-1, 1) -> [B, 1]
        # topk_indices -> [B, 5]
        # eq -> [B, 5] -> sum -> [B] (是0或1)
        r5 = (targets.view(-1, 1) == topk_indices).sum(dim=1).float().mean()

        return r1, r5

    # def configure_optimizers(self):
    #     # 1. 获取超参数
    #     weight_decay = self.hparams.weight_decay
    #     base_lr = self.hparams.lr
    #     layer_decay = self.hparams.layer_decay
    #
    #     # 2. 初始化参数分组注册表
    #     # Key: (learning_rate, weight_decay) -> Value: List[nn.Parameter]
    #     # 这样可以将相同配置的参数合并到一个 group 中
    #     grouped_parameters = {}
    #
    #     def add_param_to_group(param, lr, wd):
    #         key = (lr, wd)
    #         if key not in grouped_parameters:
    #             grouped_parameters[key] = []
    #         grouped_parameters[key].append(param)
    #
    #     # 辅助函数：判断是否跳过 Weight Decay (Bias, Norm, Scale)
    #     def check_no_wd(name, param):
    #         return (param.ndim < 2) or ("bias" in name) or ("norm" in name) or ("logit_scale" in name) or (
    #                 "gamma" in name)
    #
    #     visited_params = set()  # 记录已处理的参数 ID
    #
    #     # -------------------------------------------------------
    #     # A. Image Backbone (AstroDINO)
    #     # -------------------------------------------------------
    #     img_backbone = self.image_encoder.image_backbone
    #     # 动态获取层数
    #     # 假设 blocks 是 ModuleList 或 Sequential，长度即为 Transformer 层数 (如 12)
    #     num_layers_img = len(img_backbone.blocks)
    #
    #     for name, p in img_backbone.named_parameters():
    #         # if not p.requires_grad: continue
    #         visited_params.add(id(p))
    #
    #         # 解析层号 (Layer ID)
    #         # 目标：底层=0, blocks=1~N, norm=N+1
    #         if "patch_embed" in name or "pos_embed" in name or "cls_token" in name or "mask_token" in name:
    #             layer_id = 0
    #         elif "blocks" in name:
    #             # 键名格式: blocks.0.0.xxx ... blocks.3.11.xxx
    #             # 无论前面是 stage 几，第三段 (index 2) 始终是全局层号 (0-11)
    #             try:
    #                 layer_id = int(name.split(".")[2]) + 1  # 映射到 1 ~ 12
    #             except (IndexError, ValueError):
    #                 layer_id = 0
    #         elif "norm" in name:
    #             layer_id = num_layers_img + 1  # Top Layer
    #         else:
    #             layer_id = 0
    #
    #         # 计算 LLRD
    #         # 公式：LR = base_lr * (decay ^ (Total_Depth - Current_Depth))
    #         # 越深层 (Current_Depth 大)，指数越小，LR 越大
    #         exponent = (num_layers_img + 1) - layer_id
    #         lr = base_lr * (layer_decay ** exponent)
    #         wd = 0.0 if check_no_wd(name, p) else weight_decay
    #
    #         add_param_to_group(p, lr, wd)
    #
    #     # -------------------------------------------------------
    #     # B. Spectrum Backbone (SpecFormer)
    #     # -------------------------------------------------------
    #     spec_backbone = self.spectrum_encoder.backbone
    #     # 动态获取层数
    #     num_layers_spec = len(spec_backbone.blocks)
    #
    #     for name, p in spec_backbone.named_parameters():
    #         # if not p.requires_grad: continue
    #         visited_params.add(id(p))
    #
    #         # 解析层号
    #         if "data_embed" in name or "position_embed" in name:
    #             layer_id = 0
    #         elif "blocks" in name:
    #             # 键名格式: blocks.0.xxx ... blocks.5.xxx
    #             try:
    #                 layer_id = int(name.split(".")[1]) + 1  # 映射到 1 ~ 6
    #             except (IndexError, ValueError):
    #                 layer_id = 0
    #         elif "final_layernorm" in name:
    #             layer_id = num_layers_spec + 1  # Top Layer
    #         else:
    #             layer_id = 0
    #
    #         exponent = (num_layers_spec + 1) - layer_id
    #         lr = base_lr * (layer_decay ** exponent)
    #         wd = 0.0 if check_no_wd(name, p) else weight_decay
    #
    #         add_param_to_group(p, lr, wd)
    #
    #     # -------------------------------------------------------
    #     # C. Photometry Backbone (MLP)
    #     # -------------------------------------------------------
    #     photo_backbone = self.image_encoder.photo_backbone
    #     # 视为中间层特征，给予固定比例的学习率 (例如相当于 Image Backbone 第6层的衰减)
    #     photo_lr = base_lr * (layer_decay ** (num_layers_img // 2))
    #
    #     for name, p in photo_backbone.named_parameters():
    #         # if not p.requires_grad: continue
    #         visited_params.add(id(p))
    #
    #         wd = 0.0 if check_no_wd(name, p) else weight_decay
    #         add_param_to_group(p, photo_lr, wd)
    #
    #     # -------------------------------------------------------
    #     # D. Heads & The Rest (Projectors, Scale, etc.)
    #     # -------------------------------------------------------
    #     # 遍历所有参数，处理那些还没被归类的 (即 Heads)
    #     for name, p in self.named_parameters():
    #         # if not p.requires_grad: continue
    #         if id(p) in visited_params: continue  # 跳过已处理的 backbone 参数
    #
    #         # Heads 使用 Base LR
    #         wd = 0.0 if check_no_wd(name, p) else weight_decay
    #         add_param_to_group(p, base_lr, wd)
    #
    #     # 3. 构建最终的 param_groups 列表
    #     final_param_groups = []
    #     # 按 LR 从小到大排序 (可选，方便调试观察)
    #     sorted_keys = sorted(grouped_parameters.keys(), key=lambda x: x[0])
    #
    #     for (lr, wd) in sorted_keys:
    #         params = grouped_parameters[(lr, wd)]
    #         final_param_groups.append({
    #             "params": params,
    #             "lr": lr,
    #             "weight_decay": wd
    #         })
    #
    #     for g in final_param_groups:
    #         g["foreach"] = True
    #
    #     # 4. 初始化优化器
    #     optimizer = torch.optim.AdamW(final_param_groups)
    #
    #     # 5. 初始化 Scheduler
    #
    #     # 注意 T_max 计算: Epochs * Steps_Per_Epoch
    #     # 建议动态获取 steps_per_epoch，或者估算 (比如 670k / 256 ≈ 2617)
    #     steps_per_epoch = 283
    #     total_steps = self.hparams.epochs * steps_per_epoch
    #
    #     scheduler = CosineAnnealingWithWarmupLR(
    #         optimizer,
    #         T_max=total_steps,
    #         T_warmup=int(total_steps * 0.1),  # 10% Warmup
    #         eta_min=self.hparams.eta_min
    #     )
    #
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #         },
    #     }


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


class PhotoGuidedFusionHead(nn.Module):
    def __init__(
            self,
            img_dim: int = 384,  # 图像编码器输出维度 (ViT-Small)
            photo_dim: int = 128,  # 测光编码器输出维度
            num_photo_tokens: int = 6,  # 测光 Token 数量 (1 CLS + 5 Bands)
            latent_dim: int = 256,  # 最终对齐的潜空间维度 (Sweet Spot)
            n_head: int = 8,  # Attention 头数
            dropout: float = 0.1,  # Dropout 率
            ffn_mult: int = 4  # FFN 膨胀倍数
    ):
        """
        测光引导的图像特征提取头 (Photometry-Guided Image Extraction Head).

        逻辑:
        1. 测光 Tokens 作为 Query (主动探针)。
        2. 图像 Tokens 作为 Key/Value (被动资源池)。
        3. 通过 Cross-Attention，物理信息(测光)去图像中"吸取"对应的视觉特征。
        """
        super().__init__()

        # 1. 测光特征投影层 (128 -> 384)
        # 将测光数据的语义空间映射到图像语义空间
        self.photo_proj = nn.Linear(photo_dim, img_dim)
        self.photo_norm = nn.LayerNorm(img_dim)

        # 2. Cross-Attention 模块
        # Query = Photo, Key = Image, Value = Image
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(img_dim)

        # 3. Feed-Forward Network (FFN)
        # 标准 Transformer 结构，增强非线性特征表达
        self.ffn = nn.Sequential(
            nn.Linear(img_dim, img_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_dim * ffn_mult, img_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(img_dim)

        # 4. 最终投影头 (Projector)
        # 将融合后的 6 个 Token 展平并映射到 256 维对齐空间
        # 输入维度: 6 * 384 = 2304
        flatten_dim = num_photo_tokens * img_dim

        self.projector = nn.Sequential(
            nn.Linear(flatten_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim)  # 输出 256 维
        )

        # 初始化权重 (可选，根据需要)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image_tokens: torch.Tensor, photo_tokens: torch.Tensor):
        """
        Args:
            image_tokens: [Batch, 256, 384] (ViT Patch Tokens) - 包含大量噪声
            photo_tokens: [Batch, 6, 128]   (CLS + 5 Bands) - 物理锚点
        Returns:
            latent_vec:   [Batch, 256]      (最终对齐用的 Embedding)
        """

        # --- Step 1: 准备 Query (测光) ---
        # [B, 6, 128] -> [B, 6, 384]
        # 投影并归一化，使其能与图像特征进行点积计算
        q = self.photo_proj(photo_tokens)
        q = self.photo_norm(q)

        # --- Step 2: 准备 Key/Value (图像) ---
        # k, v 直接使用图像 tokens
        k = v = image_tokens

        # --- Step 3: Cross-Attention (核心融合) ---
        # Q(Photo) 去查 K(Image)。
        # 测光 CLS 会关注图像中心；u/g/r/i/z Token 会关注特定波段响应。
        # 输出 shape: [B, 6, 384]
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)

        # --- Step 4: 残差连接 & FFN ---
        # Residual 1: 如果图像全是噪声，Attention 结果可能很弱，
        # 此时保留原始测光特征(q)作为保底。
        x = q + attn_out
        x = self.norm1(x)

        # Residual 2: FFN 处理
        x = x + self.ffn(x)
        x = self.norm2(x)

        # 此时 x 的形状为 [B, 6, 384]
        # 这 6 个向量是"注入了视觉证据的物理特征"

        # --- Step 5: 展平并投影 ---
        # 我们保留所有波段的特异性，不进行 Mean Pooling
        # [B, 6, 384] -> [B, 2304]
        x_flat = x.flatten(start_dim=1)

        # [B, 2304] -> [B, 256]
        latent_vec = self.projector(x_flat)

        return latent_vec

class ImagePhotoHead(nn.Module):
    def __init__(
            self,
            config: str,
            image_model_weights: str,  # 图像编码器路径
            photo_model_weights: str,
            save_directory: str,
            n_head: int = 4,
            clip_embed_dim: int = 1024,  # CLIP 最终维度
            image_embed_dim: int = 384,  # Image ViT dim
            photo_embed_dim: int=128,  # Photo dim
            dropout: float = 0.1,
            freeze_backbone: bool = True,
    ):
        super().__init__()

        # 1. Image Encoder (AstroDINO)
        # ... (加载代码同前) ...
        # 关键：确保 backbone 能返回 patch tokens
        # 通常 DINOv2 的 forward_features 返回 dict: {'x_norm_clstoken': ..., 'x_norm_patchtokens': ...}

        # Define DINO config
        class config:
            output_dir = save_directory
            config_file = config
            pretrained_weights = image_model_weights
            opts = []

        # Define DINO model
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to null
        self.image_backbone, _ = setup_and_build_model(config())
        sys.stdout = sys.__stdout__  # Reset stdout

        print("Pre-trained image backbone weights loaded successfully.")

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.image_backbone.parameters():
                param.requires_grad = False

        # 2. Photometry Encoder

        if photo_model_weights:
            pl_module = MaskedPhotometryModel.load_from_checkpoint(photo_model_weights)
            self.photo_backbone = pl_module.model
            print("Pre-trained photometry backbone weights loaded successfully.")

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.photo_backbone.parameters():
                param.requires_grad = False

        # 3. 实例化融合头 (使用上面的代码)
        self.fusion_head = PhotoGuidedFusionHead(
            img_dim=image_embed_dim,
            photo_dim=photo_embed_dim,
            num_photo_tokens=6,
            latent_dim=clip_embed_dim # 我们的目标维度
        )


    def forward(self, image, photometry):
        # 1. 获取图像 Patch Features
        # 假设 backbone.forward_features 返回所有 tokens
        # out: [Batch, N_patches, 384] (不包含 CLS)
        with torch.set_grad_enabled(not self.freeze_backbone):
            image = self.image_backbone.patch_embed(image)
            for blk in self.image_backbone.blocks:
                image = blk(image)
            # img_features: [Batch, 256, 384]
            img_features = self.image_backbone.norm(image)

        # 2. 获取测光 Features
        # photo_features: [Batch, 6,128]
        with torch.set_grad_enabled(not self.freeze_backbone):
            photo_features = self.photo_backbone(photometry)['features']


        # 3. 融合并输出
        # [B, 256]
        embedding = self.fusion_head(img_features, photo_features)

        return embedding


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
        checkpoint = torch.load(model_path,map_location='cpu')
        self.backbone = SpecFormer(**checkpoint["hyper_parameters"]['init_args'])
        print("Pre-trained spectrum backbone weights loaded successfully.")

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
