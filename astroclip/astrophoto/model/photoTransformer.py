import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from astroclip.astrophoto.model.rtdl_revisiting_models import FTTransformer


# ================================================================= #
#  Part 1: The Photometry Encoder Model (FTTransformer Wrapper)     #
# ================================================================= #

class PhotometryEncoder(nn.Module):
    """
    Wrapper around FTTransformer to act as an encoder for continuous photometry data.
    """

    def __init__(
            self,
            input_dim: int = 19,
            embedding_dim: int = 768,
            n_blocks: int = 3
    ):
        """
        Args:
            input_dim (int): Number of continuous features (e.g., 15 for 5 bands * 3 props).
            embedding_dim (int): Dimension of the output embedding (e.g., 768).
            n_blocks (int): Number of Transformer blocks (depth). Recommended: 1-6.
        """
        super().__init__()

        # 1. 获取 FTTransformer 的默认推荐参数
        # 这些参数经过原作者在多个数据集上的调优，通常比随机设置更好
        kwargs = FTTransformer.get_default_kwargs(n_blocks=n_blocks)

        # 2. 覆盖默认参数以适配我们的任务
        # d_out 设为 embedding_dim，这样模型输出就是我们想要的 768 维向量
        kwargs['d_out'] = embedding_dim

        # 3. 初始化 FTTransformer
        # n_cont_features: 连续特征数量
        # cat_cardinalities: 类别特征的基数列表 (这里为空，因为全是连续值)
        self.model = FTTransformer(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (batch_size, input_dim). Continuous data.
        Returns:
            torch.Tensor: Shape (batch_size, embedding_dim).
        """
        # FTTransformer 的 forward 接收 (x_cont, x_cat)
        # 因为我们没有类别特征，第二个参数传 None
        return self.model(x, None)


# ================================================================= #
#  Part 2: The Masked Feature Modeling (MFM) Pre-training Wrapper   #
# ================================================================= #

class MFMPretrainer(L.LightningModule):
    """
    MFM Pre-training using FTTransformer backbone.
    """

    def __init__(self,
                 input_dim: int = 19,
                 embedding_dim: int = 768,
                 masking_ratio: float = 0.4,  # 掩码比例
                 learning_rate: float = 1e-4,
                 n_blocks: int = 3):  # Transformer 层数
        """
        Args:
            input_dim (int): Input feature dimension.
            embedding_dim (int): Output embedding dimension.
            masking_ratio (float): Probability of masking a feature.
            learning_rate (float): Optimizer learning rate.
            n_blocks (int): Depth of the FTTransformer.
        """
        super().__init__()
        self.save_hyperparameters()

        # 使用 FTTransformer 作为编码器
        self.encoder = PhotometryEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            n_blocks=n_blocks
        )

        # 预测头：从 Embedding 空间映射回原始特征空间
        # 这里的输入是 embedding_dim，输出是 input_dim
        self.prediction_head = nn.Linear(embedding_dim, input_dim)

    def _forward_and_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Core logic for MFM: Mask -> Encode -> Reconstruct -> Loss
        """
        # batch shape: (B, 15)

        # 1. 创建随机掩码 (Boolean mask)
        # mask 为 True 的地方表示该特征被掩盖
        mask = torch.rand(batch.shape, device=self.device) < self.hparams.masking_ratio

        # 保证至少有一个特征被掩盖，避免 Loss 变成 NaN
        if not mask.any():
            rand_idx = torch.randint(0, batch.shape[1], (1,)).item()
            mask[0, rand_idx] = True

        # 2. 创建受损输入 (Corrupted Input)
        # 对于 FTTransformer，将连续值置为 0 是一种标准的简单掩码方式
        # (注意：如果 0 在你的数据中有特殊物理含义且非常常见，可以考虑用数据均值代替)
        corrupted_x = batch.clone()
        corrupted_x[mask] = 0.0

        # 3. 编码器前向传播
        # FTTransformer 内部会将特征 Tokenize 化，加上 CLS Token 并进行 Attention
        embedding = self.encoder(corrupted_x)  # Output: (B, 768)

        # 4. 预测原始特征
        predicted_features = self.prediction_head(embedding)  # Output: (B, 15)

        # 5. 仅计算被掩盖部分的损失 (MSE)
        # 这样迫使模型利用未掩盖的波段/误差信息推断被掩盖的部分，学习波段间的物理关联
        loss = F.mse_loss(predicted_features[mask], batch[mask])

        return loss

    def training_step(self, batch, batch_idx):
        # 假设 DataLoader 返回的直接是 Tensor (B, 15)
        loss = self._forward_and_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_and_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # 使用 FTTransformer 论文推荐的优化器配置逻辑 (通常 AdamW 配合特定的 weight decay)
        # 这里为了简便，我们直接使用 encoder 内部生成的参数组 (如果有的话)
        # 或者直接使用标准的 AdamW

        # 方案 A: 使用 FTTransformer 自带的优化器创建逻辑 (推荐)
        # 下面的 make_default_optimizer 会根据参数名自动处理 weight decay (对 bias 和 LayerNorm 不使用 decay)
        # 但它默认 lr=1e-4, weight_decay=1e-5，我们需要覆盖 lr

        # 为了更灵活地控制 prediction_head 的参数，我们手动构建参数组
        encoder_groups = self.encoder.model.make_parameter_groups()

        # 将 prediction_head 的参数加入到 main_group (有 weight decay) 中
        head_params = list(self.prediction_head.parameters())
        encoder_groups[0]['params'].extend(head_params)  # type: ignore

        optimizer = torch.optim.AdamW(
            encoder_groups,
            lr=self.hparams.learning_rate,
            weight_decay=1e-5
        )

        return optimizer


# ================================================================= #
#  Example Usage                                                    #
# ================================================================= #
if __name__ == "__main__":
    # 模拟 15D 类星体测光数据
    # 例如: u, g, r, i, z (5 bands) * (psfMag, extinction, psfMagErr) = 15 features
    batch_size = 32
    input_dim = 19
    dummy_data = torch.randn(batch_size, input_dim)

    # 实例化模型
    model = MFMPretrainer(
        input_dim=input_dim,
        embedding_dim=256,
        learning_rate=1e-4,
        n_blocks=3
    )

    # 简单测试一次前向传播
    model.eval()
    with torch.no_grad():
        loss = model._forward_and_loss(dummy_data)
        print(f"Test Pass - Loss: {loss.item()}")

        # 测试编码器输出形状
        emb = model.encoder(dummy_data)
        print(f"Embedding Shape: {emb.shape}")  # 应该是 (32, 768)