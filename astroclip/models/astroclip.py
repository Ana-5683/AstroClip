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
            lr: float = 1e-4,
            weight_decay: float = 0.05,
            epochs: int = 100,
            eta_min: float = 5e-7,
            logit_scale: float = 15.5,
            learnable_logit_scale: bool = False,
            layer_decay: float = 0.7,  # LLRD 衰减系数
            loss_type: str = "clip",
            temperature: float = 0.07,
            target_temperature: float =0.04,
            cuda_gather: bool = True,
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
        if loss_type == "spec":
            self.criterion = SpectrumWeightedInfoNCELoss(temperature,  target_temperature)
        else:
            self.criterion = CLIPLoss(loss_type=loss_type)

        self.cuda_gather=cuda_gather
        if self.cuda_gather:
            print(f"开启显卡聚合")
        else:
            print(f"关闭显卡聚合")

    def forward(
            self,
            input: torch.Tensor,
            input_extra: torch.Tensor = None,  # 测光
            input_type: str="ip",
    ):
        if input_type == "ip":
            return self.image_encoder(input,input_extra)

        elif input_type == "spectrum":
            return self.spectrum_encoder(input)

        else:
            raise ValueError("Input type must be either 'image' or 'spectrum'")

    def training_step(self, batch, batch_idx):
        im, sp , photo= batch["image"], batch["spectrum"] , batch["params"]

        # 1. 限制 Logit Scale 范围 (防止梯度爆炸)
        # Max: ln(100) ≈ 4.605
        if isinstance(self.logit_scale, nn.Parameter):
            with torch.no_grad():
                self.logit_scale.clamp_(max=4.6052)

        # 2. 获取实际 Scale (Exp)
        # 注意：这里我们取 exp，让 Scale 回到 15.5~100 的线性空间
        actual_scale = self.logit_scale.exp()

        # 1. 获取本地特征 (不进行归一化，Raw Features)
        local_image_features = self.image_encoder(im,photo)
        local_spectrum_features = self.spectrum_encoder(sp)

        # 2. 跨显卡聚合特征
        if self.cuda_gather:
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
        im, sp , photo= batch["image"], batch["spectrum"] , batch["params"]

        # 1. 获取本地特征 (不进行归一化，Raw Features)
        local_image_features = self.image_encoder(im,photo)
        local_spectrum_features = self.spectrum_encoder(sp)

        # 2. 跨显卡聚合特征 (为了保持与训练时一致的负样本数量)
        if self.cuda_gather:
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
        all_img_feats = F.normalize(all_img_feats, dim=-1, eps=1e-3)
        all_spec_feats = F.normalize(all_spec_feats, dim=-1, eps=1e-3)

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
    #     # 1. 获取基础超参数
    #     weight_decay = self.hparams.weight_decay
    #     base_lr = self.hparams.lr
    #     layer_decay = self.hparams.layer_decay
    #
    #
    #     # 临时存储分组参数的字典
    #     # Key: (lr_multiplier, wd_multiplier)
    #     # Value: List[Tuple[str, nn.Parameter]] -> 存入 (name, param) 以便验证
    #     grouped_parameters = {}
    #
    #     def add_param_to_group(name, param, lr_mult, wd_mult):
    #         key = (lr_mult, wd_mult)
    #         if key not in grouped_parameters:
    #             grouped_parameters[key] = []
    #         # 这里将 name 一并加入
    #         grouped_parameters[key].append((name, param))
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
    #     num_layers_img = img_backbone.n_blocks
    #
    #     for name, p in img_backbone.named_parameters():
    #         if not p.requires_grad: continue
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
    #         # 1.2 计算 Multiplier
    #         # LR Multiplier: layer_decay ^ (Total - Current)
    #         lr_mult = layer_decay ** ((num_layers_img + 1) - layer_id)
    #         # WD Multiplier: 0.0 or 1.0
    #         wd_mult = 0.0 if check_no_wd(name, p) else 1.0
    #
    #         add_param_to_group(name,p, lr_mult, wd_mult)
    #
    #     # -------------------------------------------------------
    #     # B. Spectrum Backbone (SpecFormer)
    #     # -------------------------------------------------------
    #     spec_backbone = self.spectrum_encoder.backbone
    #     # 动态获取层数
    #     num_layers_spec = len(spec_backbone.blocks)
    #
    #     for name, p in spec_backbone.named_parameters():
    #         if not p.requires_grad: continue
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
    #         # 2.2 计算 Multiplier
    #         lr_mult = layer_decay ** ((num_layers_spec + 1) - layer_id)
    #         wd_mult = 0.0 if check_no_wd(name, p) else 1.0
    #
    #         add_param_to_group(name,p, lr_mult, wd_mult)
    #
    #     # -------------------------------------------------------
    #     # C. Photometry Backbone (MLP)
    #     # -------------------------------------------------------
    #     photo_backbone = self.image_encoder.photo_backbone
    #     # 视为中间层特征，给予固定比例的学习率 (例如相当于 Image Backbone 第6层的衰减)
    #     lr_mult = layer_decay ** (num_layers_img // 2)
    #
    #     for name, p in photo_backbone.named_parameters():
    #         # if not p.requires_grad: continue
    #         visited_params.add(id(p))
    #
    #         wd_mult = 0.0 if check_no_wd(name, p) else 1.0
    #         add_param_to_group(name,p, lr_mult, wd_mult)
    #
    #     # -------------------------------------------------------
    #     # D. Heads & The Rest (Projectors, Scale, etc.)
    #     # -------------------------------------------------------
    #     # 遍历所有参数，处理那些还没被归类的 (即 Heads)
    #     for name, p in self.named_parameters():
    #         if not p.requires_grad: continue
    #         if id(p) in visited_params: continue  # 跳过已处理的 backbone 参数
    #
    #         # Heads 不进行 Layer Decay，LR Multiplier 为 1.0
    #         lr_mult = 1.0
    #         wd_mult = 0.0 if check_no_wd(name, p) else 1.0
    #
    #         add_param_to_group(name,p, lr_mult, wd_mult)
    #
    #     # =======================================================
    #     # Part 4: 构建最终的 optimizer_grouped_parameters
    #     # =======================================================
    #     final_param_groups = []
    #
    #     # 按照 lr_multiplier 从小到大排序 (保持确定性，便于调试)
    #     sorted_keys = sorted(grouped_parameters.keys(), key=lambda x: x[0])
    #
    #     for (lr_mult, wd_mult) in sorted_keys:
    #         # 取出列表 [(name, param), (name, param), ...]
    #         name_param_list = grouped_parameters[(lr_mult, wd_mult)]
    #
    #         # 分离 name 和 param
    #         # params 用于优化器，names 用于验证/打印
    #         params = [p for (n, p) in name_param_list]
    #         names = [n for (n, p) in name_param_list]
    #
    #         # 计算最终 LR 和 WD
    #         final_lr = base_lr * lr_mult
    #         final_wd = weight_decay * wd_mult
    #
    #         # --- 验证打印 (可选) ---
    #         # print(f"Group: LR_mult={lr_mult:.4f}, WD_mult={wd_mult}, Final_LR={final_lr:.2e}")
    #         # print(f"  -> Contains {len(params)} params: {names[:3]} ...")
    #         # ---------------------
    #
    #         final_param_groups.append({
    #             "params": params,
    #             "lr": final_lr,
    #             "weight_decay": final_wd,
    #             # 如果需要验证，可以将 names 也存在这里，优化器会自动忽略多余的键，或者你在return前pop掉
    #             # "param_names": names
    #         })
    #
    #     # 5. 初始化优化器
    #     optimizer = torch.optim.AdamW(final_param_groups)
    #
    #     # 6. 初始化 Scheduler
    #     # 注意：Scheduler 会基于 Optimizer 中每个 group 设定的 initial_lr 进行衰减
    #     scheduler = CosineAnnealingWithWarmupLR(
    #         optimizer,
    #         T_max=28_400,
    #         T_warmup=2_840,
    #         # T_max=14_200,
    #         # T_warmup=1_420,
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


class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None, eps=np.log(1e-45)):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.eps = eps

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) * self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) * self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * self.eps, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma,
                                                                                dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)


class CLIPLoss(nn.Module):
    def __init__(self, loss_type: str = "clip", dcl_sigma: float = 0.5):
        """
        Args:
            loss_type: 'clip', 'dcl', or 'dclw'
            dcl_sigma: Sigma for DCLW
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.dcl_sigma = dcl_sigma

        if self.loss_type not in ["clip", "dcl", "dclw"]:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def get_logits(
            self,
            image_features: torch.FloatTensor,
            spectrum_features: torch.FloatTensor,
            logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize features
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)

        # Calculate the logits
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        return logits_per_image, logits_per_image.T

    def forward(
            self,
            image_features: torch.FloatTensor,
            spectrum_features: torch.FloatTensor,
            logit_scale: float,
            output_dict: bool = False,
    ) -> torch.FloatTensor:

        # 1. Standard InfoNCE (CLIP) Logic
        if self.loss_type == "clip":
            logits_per_image, logits_per_spectrum = self.get_logits(
                image_features, spectrum_features, logit_scale
            )
            labels = torch.arange(
                logits_per_image.shape[0], device=image_features.device, dtype=torch.long
            )
            total_loss = (
                                 F.cross_entropy(logits_per_image, labels)
                                 + F.cross_entropy(logits_per_spectrum, labels)
                         ) / 2

        # 2. DCL / DCLW Logic
        else:
            # DCL 要求输入必须是 Normalized 的
            image_features = F.normalize(image_features, dim=-1, eps=1e-3)
            spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)

            # 实例化 DCL 或 DCLW
            # 注意：由于 temperature 可能是动态变化的（如果是 learnable），
            # 我们需要在 forward 中实例化，或者修改 DCL 类支持动态 temp。
            # 这里选择最安全的方式：每步实例化 (开销极小)
            if self.loss_type == "dcl":
                criterion = DCL(temperature=logit_scale)
            elif self.loss_type == "dclw":
                criterion = DCLW(temperature=logit_scale, sigma=self.dcl_sigma)

            # 计算双向 Loss
            # DCL 实现计算的是 z1 对 z2 的 Loss (包含 z1 自身的负样本)
            loss_i2s = criterion(image_features, spectrum_features)
            loss_s2i = criterion(spectrum_features, image_features)

            total_loss = (loss_i2s + loss_s2i) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class SpectrumWeightedInfoNCELoss(nn.Module):
    def __init__(self, temperature, target_temperature):
        """
        Args:
            temperature (float): 控制预测 logits 的平滑程度 (Student)。
            target_temperature (float): 控制目标分布的锐度 (Teacher)。
                                        通常设得比 temperature 小一点，使目标分布更尖锐。
        """
        super().__init__()
        self.temperature = temperature
        self.target_temperature = target_temperature

    def forward(self, image_features, spectrum_features, logit_scale: float):
        """
        Args:
            image_features: (batch_size, embed_dim) - 来自 Image Projection Head
            spectrum_features:  (batch_size, embed_dim) - 来自 Spec Projection Head
        """

        # 1. 归一化特征 (L2 Normalize)
        # InfoNCE 依赖余弦相似度，必须先做归一化
        image_norm = F.normalize(image_features, dim=-1, eps=1e-3)
        spec_norm = F.normalize(spectrum_features, dim=-1, eps=1e-3)

        # ==========================================================
        # 2. 构建 Soft Target (教师信号：光谱指导)
        # ==========================================================
        # 计算光谱之间的自相似度矩阵 (Batch_size x Batch_size)
        # 如果 spec[i] 和 spec[j] 很像，这个值会接近 1
        with torch.no_grad():
            spec_sim_matrix = torch.matmul(spec_norm, spec_norm.T)

            # 对角线 mask (可选): 标准 InfoNCE 会 Mask 掉对角线，
            # 但在 Soft Target 中，对角线代表"自己像自己"，通常保留并作为概率最高的类。

            # 使用 Softmax 生成目标概率分布
            # 这里除以 target_temperature 是为了控制分布的"熵"
            # 温度越低，分布越接近 One-hot；温度越高，允许更多相似样本被视为正样本
            targets = F.softmax(spec_sim_matrix / self.target_temperature, dim=1)

        # ==========================================================
        # 3. 计算 Prediction Logits (学生预测)
        # ==========================================================
        # 计算图像和光谱的交叉相似度
        logits = torch.matmul(image_norm, spec_norm.T)
        logits = logits / self.temperature

        # ==========================================================
        # 4. 计算 Loss (Cross Entropy with Soft Targets)
        # ==========================================================
        # 公式: Loss = - sum( Target * log(Predicted_Prob) )
        # LogSoftmax 在数值上比 log(softmax) 更稳定
        log_probs = F.log_softmax(logits, dim=1)

        # 计算逐样本的 Cross Entropy
        loss = -torch.sum(targets * log_probs, dim=1).mean()

        # 你也可以选择做双向 Loss (Image->Spec 和 Spec->Image)
        # 但考虑到你的任务是 Image->Redshift，单向对齐通常足够，或者加权双向

        return loss

class PhotoGuidedFusionHead_qp(nn.Module):
    def __init__(
            self,
            img_dim: int = 384,  # 图像编码器输出维度 (ViT-Small)
            photo_dim: int = 128,  # 测光编码器输出维度
            num_photo_tokens: int = 5,  # 测光 Token 数量 (1 CLS + 5 Bands)
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
        # 将融合后的 5 个 Token 展平并映射到 256 维对齐空间
        # 输入维度: 5 * 384 = 2304
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

        # 输出 shape: [B, 5, 384]
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

class PhotoGuidedFusionHead_qi(nn.Module):
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

        # 3. 维度适配层 (关键修改)
        # 将图像从 384 升维到 768
        self.image_proj = nn.Identity()

        self.photo_proj = nn.Linear(photo_dim, img_dim)  # 或者 nn.Linear(768, 768) 进行微调
        self.photo_norm = nn.LayerNorm(img_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,  # 384
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(img_dim)
        self.ffn = MLP(
            in_features=img_dim,
            hidden_features=4 * img_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(img_dim)

        # 4. Final Projector (CLIP 空间)
        self.clip_projector = CrossAttentionHead(
            embed_dim=latent_dim,
            n_head=n_head,
            model_embed_dim=img_dim,
            dropout=dropout
        )

        # Set up MLP
        self.mlp_final = MLP(
            in_features=latent_dim,
            hidden_features=4 * latent_dim,
            dropout=dropout,
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
        photo_features = self.photo_proj(photo_tokens)
        photo_features = self.photo_norm(photo_features)

        # --- Step 2: 准备 Key/Value (图像) ---
        # k, v 直接使用图像 tokens
        img_features=self.image_proj (image_tokens)

        # Cross Attention: Q=Image, K=Photo, V=Photo
        # 这一步：图像看着测光数据，更新自己的特征
        attn_out, _ = self.cross_attn(
            query=img_features,
            key=photo_features,
            value=photo_features
        )

        # 4. 残差连接 (现在维度匹配了，都是768)
        # img_features (768) + attn_out (768)
        # 物理意义：保留升维后的图像形态 + 测光修正
        img_features = self.norm1(img_features + attn_out)

        # FFN [batch,768,768]
        img_features = self.norm2(img_features + self.ffn(img_features))

        # 5. 最终投影
        # 输入 768 -> 输出 1024 [batch,1,1024]
        x, attentions = self.clip_projector(img_features)
        x = x + self.mlp_final(x)

        return x.squeeze()


class PhotoGuidedFusionHeadConc(nn.Module):
    def __init__(
            self,
            img_dim: int = 384,  # 图像编码器输出维度 (ViT-Small)
            photo_dim: int = 128,  # 测光编码器输出维度
            latent_dim: int = 256,  # 最终对齐的潜空间维度 (Sweet Spot)
            n_head: int = 8,  # Attention 头数
            dropout: float = 0.1,  # Dropout 率
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

        self.image_norm = nn.LayerNorm(img_dim)

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=latent_dim,
            n_head=n_head,
            model_embed_dim=img_dim,
            dropout=dropout,
        )

        # Set up MLP
        self.mlp = MLP(
            in_features=latent_dim,
            hidden_features=4 * latent_dim,
            dropout=dropout,
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
        # [B, 5, 128] -> [B, 5, 384]
        # 投影并归一化，使其能与图像特征进行点积计算
        photo_feature = self.photo_proj(photo_tokens)
        photo_feature = self.photo_norm(photo_feature)

        # --- Step 2: 准备 Key/Value (图像) ---
        # k, v 直接使用图像 tokens
        image_feature =self.image_norm(image_tokens)

        # contact
        x = torch.concatenate([photo_feature, image_feature],dim=1)

        # Pass through cross-attention
        x, attentions = self.cross_attention(x)

        # Pass through MLP and residual connection
        x = x + self.mlp(x)

        return x.squeeze()




class ImagePhotoHead(nn.Module):
    def __init__(
            self,
            config: str,
            image_model_weights: str,  # 图像编码器路径
            photo_model_weights: str,
            save_directory: str,
            fusion_mode: str,
            n_head: int = 4, #TODO: 4
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

        # 3. 实例化融合头
        if fusion_mode== "p2i":
            self.fusion_head=PhotoGuidedFusionHead_P2I(
                img_dim=image_embed_dim,
                photo_dim=photo_embed_dim,
                num_photo_tokens=6,
                dropout=dropout,
                n_head=n_head,
                latent_dim=clip_embed_dim # 我们的目标维度
            )


        elif fusion_mode == "i2p":

            self.fusion_head=PhotoGuidedFusionHead_I2P(
                img_dim=image_embed_dim,
                photo_dim=photo_embed_dim,
                num_photo_tokens=6,
                dropout=dropout,
                n_head=n_head,
                latent_dim=clip_embed_dim # 我们的目标维度
            )
        elif fusion_mode == "bi":
            self.fusion_head=PhotoGuidedFusionHead_Bi(
                img_dim=image_embed_dim,
                photo_dim=photo_embed_dim,
                num_photo_tokens=6,
                dropout=dropout,
                n_head=n_head,
                latent_dim=clip_embed_dim # 我们的目标维度
            )

        elif fusion_mode == "conc":
            self.fusion_head=PhotoGuidedFusionHeadConc(
                img_dim=image_embed_dim,
                photo_dim=photo_embed_dim,
                dropout=dropout,
                n_head=n_head,
                latent_dim=clip_embed_dim # 我们的目标维度
            )

        elif fusion_mode == "q_i":
            self.fusion_head=PhotoGuidedFusionHead_qi(
                img_dim=image_embed_dim,
                photo_dim=photo_embed_dim,
                dropout=dropout,
                n_head=n_head,
                latent_dim=clip_embed_dim # 我们的目标维度
            )
        elif fusion_mode == "q_p":
            self.fusion_head=PhotoGuidedFusionHead_qp(
                img_dim=image_embed_dim,
                photo_dim=photo_embed_dim,
                dropout=dropout,
                n_head=n_head,
                latent_dim=clip_embed_dim # 我们的目标维度
            )
        else:
            raise ValueError(f"Invalid fusion mode: {fusion_mode}")


        # self.fusion_head = PhotoGuidedFusionHead(
        #     img_dim=image_embed_dim,
        #     photo_dim=photo_embed_dim,
        #     num_photo_tokens=6,
        #     latent_dim=clip_embed_dim # 我们的目标维度
        # )


    def forward(self, image, photometry):
        # 1. 获取图像 Patch Features
        # 假设 backbone.forward_features 返回所有 tokens
        # out: [Batch, N_patches, 384] (不包含 CLS)
        with torch.set_grad_enabled(not self.freeze_backbone):
            image = self.image_backbone.patch_embed(image)
            for blk in self.image_backbone.blocks:
                image = blk(image)
            img_features = self.image_backbone.norm(image)

        # 2. 获取测光 Features
        # photo_features: [Batch, 6,128]
        with torch.set_grad_enabled(not self.freeze_backbone):
            photo_features = self.photo_backbone(photometry)['features']
            # 去除cls token
            photo_features=photo_features[:,1:]


        # 3. 融合并输出
        # [B, 256]
        embedding = self.fusion_head(img_features, photo_features)

        return embedding

class PhotoGuidedFusionHead_P2I(nn.Module):
    def __init__(
            self,
            img_dim: int = 384,  # 图像编码器输出维度 (ViT-Small)
            photo_dim: int = 128,  # 测光编码器输出维度
            num_photo_tokens: int = 6,  # 测光 Token 数量 (1 CLS + 5 Bands)
            latent_dim: int = 256,  # 最终对齐的潜空间维度
            n_head: int = 8,  # Attention 头数
            dropout: float = 0.1,  # Dropout 率
            ffn_mult: int = 4  # FFN 膨胀倍数
    ):
        """
        P2I (Photometry-to-Image) Fusion Head
        逻辑: 物理驱动的视觉特征提取。测光作为 Query，去图像 Patch 中搜索相关特征。
        """
        super().__init__()

        # ----------------------------------------------------------------
        # 1. 维度适配器 (Adapter)
        # ----------------------------------------------------------------
        # 将测光特征从 128 维提升到 384 维，以便与图像特征进行交互
        self.photo_proj = nn.Sequential(
            nn.Linear(photo_dim, img_dim),
            nn.LayerNorm(img_dim)
        )

        # ----------------------------------------------------------------
        # 2. Cross-Attention 组件 (Pre-Norm 结构)
        # ----------------------------------------------------------------
        # Query Norm: 对测光特征归一化
        self.norm_q = nn.LayerNorm(img_dim)

        # Context Norm: 对图像特征归一化 (Key/Value)
        self.norm_kv = nn.LayerNorm(img_dim)

        # 多头注意力: Q=Photo, K=Image, V=Image
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        # ----------------------------------------------------------------
        # 3. Feed-Forward Network (FFN) 组件
        # ----------------------------------------------------------------
        self.norm_ffn = nn.LayerNorm(img_dim)

        self.ffn = nn.Sequential(
            nn.Linear(img_dim, img_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_dim * ffn_mult, img_dim),
            nn.Dropout(dropout)
        )

        # ----------------------------------------------------------------
        # 4. 最终投影头 (Projector)
        # ----------------------------------------------------------------
        # 输入维度计算:
        # P2I 模式输出为 [Batch, 6, 384]。
        # 为了保留 u,g,r,i,z 的波段特异性，我们将其展平 (Flatten)。
        # Input Dim = 6 * 384 = 2304
        flatten_dim = num_photo_tokens * img_dim

        self.projector = nn.Sequential(
            nn.Linear(flatten_dim, flatten_dim),  # 可选: 先过一层线性变换
            nn.LayerNorm(flatten_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flatten_dim, latent_dim)  # 映射到最终对齐空间
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier 初始化，利于 Transformer 收敛
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image_features: torch.Tensor, photo_features: torch.Tensor):
        """
        Args:
            image_features: [Batch, 257, 384] (ViT Output: 1 CLS + 256 Patches)
            photo_features: [Batch, 6, 128]   (Photo Output: 1 CLS + 5 Bands)
        Returns:
            latent_vec:     [Batch, latent_dim]
        """

        # --- Step 1: 维度对齐与 Query/Key 准备 ---

        # 1.1 测光投影: [B, 6, 128] -> [B, 6, 384]
        # 这是我们的 Query (Q)
        q = self.photo_proj(photo_features)

        # 1.2 图像处理: [B, 257, 384] -> [B, 256, 384]
        # 去除 CLS token，只保留 Patch tokens
        # 这是我们的 Key (K) 和 Value (V)
        # 理由: 让测光去关注具体的空间区域，而非图像的全局摘要
        kv = image_features[:, 1:, :]

        # --- Step 2: Cross-Attention (Pre-Norm) ---

        # 2.1 Pre-Norm
        q_norm = self.norm_q(q)  # Norm Photo
        kv_norm = self.norm_kv(kv)  # Norm Image

        # 2.2 Attention
        # attn_out: [B, 6, 384]
        # 物理含义: 注入了视觉纹理信息的测光特征
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)

        # 2.3 Residual Connection 1
        # 即使图像全是噪声，原始测光特征(q)依然保留
        x = q + attn_out

        # --- Step 3: FFN (Pre-Norm) ---

        # 3.1 Residual Connection 2
        x = x + self.ffn(self.norm_ffn(x))

        # 此时 x 的形状: [B, 6, 384]

        # --- Step 4: 聚合与输出 ---

        # 4.1 Flatten: [B, 6, 384] -> [B, 2304]
        # 保留每个波段的位置信息
        x_flat = x.flatten(start_dim=1)

        # 4.2 Project to Latent Space: [B, 2304] -> [B, latent_dim]
        latent_vec = self.projector(x_flat)

        return latent_vec

class PhotoGuidedFusionHead_I2P(nn.Module):
    def __init__(
            self,
            img_dim: int = 384,      # 图像编码器输出维度
            photo_dim: int = 128,    # 测光编码器输出维度
            latent_dim: int = 256,   # 最终对齐的潜空间维度
            num_photo_tokens: int = 6, # 忽视就好
            n_head: int = 8,         # Attention 头数
            dropout: float = 0.1,    # Dropout 率
            ffn_mult: int = 4        # FFN 膨胀倍数
    ):
        """
        I2P (Image-to-Photometry) Fusion Head
        逻辑: 视觉特征的物理增强。
        图像 CLS 作为 Query，去测光波段 (u,g,r,i,z) 中“校准”自身的亮度和颜色信息。
        """
        super().__init__()

        # ----------------------------------------------------------------
        # 1. 维度适配器 (Adapter)
        # ----------------------------------------------------------------
        # 虽然测光是 Key/Value，但维度必须与 Query (Image: 384) 一致才能计算 Attention
        # [B, 5, 128] -> [B, 5, 384]
        self.photo_proj = nn.Sequential(
            nn.Linear(photo_dim, img_dim),
            nn.LayerNorm(img_dim)
        )

        # ----------------------------------------------------------------
        # 2. Cross-Attention 组件 (Pre-Norm)
        # ----------------------------------------------------------------
        # Query Norm: 对图像 CLS 归一化
        self.norm_q = nn.LayerNorm(img_dim)

        # Context Norm: 对测光特征归一化 (Key/Value)
        self.norm_kv = nn.LayerNorm(img_dim)

        # Q=Image(CLS), K=Photo(Bands), V=Photo(Bands)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        # ----------------------------------------------------------------
        # 3. FFN 组件
        # ----------------------------------------------------------------
        self.norm_ffn = nn.LayerNorm(img_dim)

        self.ffn = nn.Sequential(
            nn.Linear(img_dim, img_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_dim * ffn_mult, img_dim),
            nn.Dropout(dropout)
        )

        # ----------------------------------------------------------------
        # 4. 最终投影头 (Projector)
        # ----------------------------------------------------------------
        # I2P 模式下，输出是单个增强后的 CLS Token
        # Input Dim = 1 * 384 = 384 (不需要 Flatten 2304)
        projector_input_dim = img_dim

        self.projector = nn.Sequential(
            nn.Linear(projector_input_dim, projector_input_dim),
            nn.LayerNorm(projector_input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projector_input_dim, latent_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image_features: torch.Tensor, photo_features: torch.Tensor):
        """
        Args:
            image_features: [Batch, 257, 384] (0 is CLS)
            photo_features: [Batch, 6, 128]   (0 is CLS, 1-5 are Bands)
        Returns:
            latent_vec:     [Batch, latent_dim]
        """

        # --- Step 1: 维度对齐与 Query/Key 准备 ---

        # 1.1 图像处理: [B, 257, 384] -> [B, 1, 384]
        # 只取 CLS Token 作为 Query (最高效)
        q = image_features[:, 0:1, :]

        # 1.2 测光处理: [B, 6, 128] -> [B, 5, 128] -> [B, 5, 384]
        # 去除 Photo CLS (只用波段数据)，并投影到 384 维
        kv_raw = photo_features[:, 1:, :]
        kv = self.photo_proj(kv_raw)

        # --- Step 2: Cross-Attention (Pre-Norm) ---

        # 2.1 Pre-Norm
        q_norm = self.norm_q(q)    # Norm Image CLS
        kv_norm = self.norm_kv(kv) # Norm Photo Bands

        # 2.2 Attention
        # attn_out: [B, 1, 384]
        # 物理含义: 吸收了测光 SED 信息的图像特征
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)

        # 2.3 Residual Connection
        x = q + attn_out

        # --- Step 3: FFN (Pre-Norm) ---
        x = x + self.ffn(self.norm_ffn(x))

        # 此时 x 的形状: [B, 1, 384]

        # --- Step 4: 聚合与输出 ---

        # 4.1 Squeeze: [B, 1, 384] -> [B, 384]
        x_squeezed = x.squeeze(1)

        # 4.2 Project: [B, 384] -> [B, latent_dim]
        latent_vec = self.projector(x_squeezed)

        return latent_vec

class PhotoGuidedFusionHead_Bi(nn.Module):
    def __init__(
            self,
            img_dim: int = 384,  # 图像编码器输出维度
            photo_dim: int = 128,  # 测光编码器输出维度
            num_photo_tokens: int = 6,  # 测光 Token 数量
            latent_dim: int = 256,  # 最终对齐的潜空间维度
            n_head: int = 8,  # Attention 头数
            dropout: float = 0.1,  # Dropout 率
            ffn_mult: int = 4  # FFN 膨胀倍数
    ):
        """
        Bidirectional (Bi) Fusion Head
        逻辑: 全互联双向融合。
        同时执行 P2I (物理找视觉) 和 I2P (视觉找物理) 两个分支，
        并将结果拼接，获得最完整的特征表达。
        """
        super().__init__()

        # =================================================================
        # 1. 共享维度适配器 (Shared Adapter)
        # =================================================================
        # 将测光特征 [B, N, 128] -> [B, N, 384]
        # 两个分支都使用投影后的测光特征
        self.photo_proj = nn.Sequential(
            nn.Linear(photo_dim, img_dim),
            nn.LayerNorm(img_dim)
        )

        # =================================================================
        # 2. Stream A: P2I Components (Photometry-to-Image)
        # =================================================================
        # Q=Photo(Full), K=Image(Patch)
        self.norm_q_p2i = nn.LayerNorm(img_dim)
        self.norm_kv_p2i = nn.LayerNorm(img_dim)

        self.attn_p2i = nn.MultiheadAttention(
            embed_dim=img_dim, num_heads=n_head, dropout=dropout, batch_first=True
        )

        self.norm_ffn_p2i = nn.LayerNorm(img_dim)
        self.ffn_p2i = self._build_ffn(img_dim, ffn_mult, dropout)

        # =================================================================
        # 3. Stream B: I2P Components (Image-to-Photometry)
        # =================================================================
        # Q=Image(CLS), K=Photo(Bands)
        self.norm_q_i2p = nn.LayerNorm(img_dim)
        self.norm_kv_i2p = nn.LayerNorm(img_dim)

        self.attn_i2p = nn.MultiheadAttention(
            embed_dim=img_dim, num_heads=n_head, dropout=dropout, batch_first=True
        )

        self.norm_ffn_i2p = nn.LayerNorm(img_dim)
        self.ffn_i2p = self._build_ffn(img_dim, ffn_mult, dropout)

        # =================================================================
        # 4. 最终投影头 (Final Projector)
        # =================================================================
        # 计算拼接后的总维度
        # Stream A (P2I) Output: [B, 6, 384] -> Flatten -> 2304
        # Stream B (I2P) Output: [B, 1, 384] -> Squeeze -> 384
        # Total: 2304 + 384 = 2688
        concat_dim = (num_photo_tokens * img_dim) + img_dim

        self.projector = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),  # 可选: 混合层
            nn.LayerNorm(concat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim, latent_dim)
        )

        self._init_weights()

    def _build_ffn(self, dim, mult, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image_features: torch.Tensor, photo_features: torch.Tensor):
        """
        Args:
            image_features: [Batch, 257, 384] (0=CLS)
            photo_features: [Batch, 6, 128]   (0=CLS)
        """

        # --- Pre-processing: 维度对齐 ---
        # [B, 6, 128] -> [B, 6, 384]
        photo_proj = self.photo_proj(photo_features)

        # =======================================================
        # Stream A: P2I (Photo Queries Image Patches)
        # =======================================================
        # Q = Photo Full [B, 6, 384]
        q_a = photo_proj
        # K, V = Image Patches [B, 256, 384]
        kv_a = image_features[:, 1:, :]

        # 1. Attention
        q_a_norm = self.norm_q_p2i(q_a)
        kv_a_norm = self.norm_kv_p2i(kv_a)
        attn_out_a, _ = self.attn_p2i(query=q_a_norm, key=kv_a_norm, value=kv_a_norm)

        # 2. Residual & FFN
        x_a = q_a + attn_out_a
        x_a = x_a + self.ffn_p2i(self.norm_ffn_p2i(x_a))

        # 3. Flatten [B, 6, 384] -> [B, 2304]
        feat_a = x_a.flatten(start_dim=1)

        # =======================================================
        # Stream B: I2P (Image CLS Queries Photo Bands)
        # =======================================================
        # Q = Image CLS [B, 1, 384]
        q_b = image_features[:, 0:1, :]
        # K, V = Photo Bands [B, 5, 384] (Remove Photo CLS)
        kv_b = photo_proj[:, 1:, :]

        # 1. Attention
        q_b_norm = self.norm_q_i2p(q_b)
        kv_b_norm = self.norm_kv_i2p(kv_b)
        attn_out_b, _ = self.attn_i2p(query=q_b_norm, key=kv_b_norm, value=kv_b_norm)

        # 2. Residual & FFN
        x_b = q_b + attn_out_b
        x_b = x_b + self.ffn_i2p(self.norm_ffn_i2p(x_b))

        # 3. Squeeze [B, 1, 384] -> [B, 384]
        feat_b = x_b.squeeze(1)

        # =======================================================
        # Fusion & Output
        # =======================================================
        # [B, 2304] + [B, 384] -> [B, 2688]
        combined = torch.cat([feat_a, feat_b], dim=1)

        # [B, 2688] -> [B, latent_dim]
        latent_vec = self.projector(combined)

        return latent_vec

class ImageHead(nn.Module):
    def __init__(
            self,
            config: str,
            model_weights: str,
            save_directory: str,
            embed_dim: int = 1024,
            n_head: int = 4,
            model_embed_dim: int = 768,
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
        checkpoint = torch.load(model_path, map_location='cpu')
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
