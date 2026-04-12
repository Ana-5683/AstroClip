# AstroCLIP 项目结构详解

AstroCLIP 是一个跨模态自监督基础模型，用于创建类星体多波段图像和光学光谱的共享嵌入空间。本文档将详细介绍项目的整体结构和各个组成部分的作用。

## 项目概述

AstroCLIP 是基于 PyTorch 实现的天文图像与光谱跨模态学习框架，主要包含三个核心模块：

1. **图像编码器 (Image Encoder)** - 基于 DINOv2 预训练的 Vision Transformer
2. **光谱编码器 (Spectrum Encoder)** - 基于掩码建模的 1D Transformer
3. **CLIP 对齐模块** - 将两种模态的嵌入空间对齐

---

## 核心目录结构

```
AstroCLIP-main/
├── astroclip/                    # 主包目录
├── dinov2/                       # DINOv2 框架实现
├── configs/                      # (astroclip、specformer)配置文件目录
├── downstream_tasks/             # 下游任务
├── dbx/                          # dbx的数据处理脚本（忽视）
├── dsm/                          # dsm的数据分析与处理
├── utils/                        # 工具脚本
├── assets/                       # 资源文件（忽视）
├── scripts/                      # 脚本文件
├── configs/                      # 配置文件
├── environment.yml               # 环境配置（忽视）
├── requirements.txt              # 依赖要求（忽视）
├── README.md                     # 项目说明文档（忽视）
└── LICENSE                       # 许可证（忽视）
```

---

## astroclip/ 主包目录

### 1. 核心模型模块 astroclip/models/

| 文件 | 作用 |
|------|------|
| astroclip.py | **主模型文件**，实现 AstroCLIP 跨模态模型，包含图像和光谱编码器及 CLIP 对齐层 |
| specformer.py | **光谱 Transformer 模型**，基于掩码建模预训练的光谱编码器 |
| astroclip_photo.py | CLIP 模型变体1 |
| astroclip_qp_ki_transformer.py | CLIP 模型变体2 |
| astroclip_qp_ki_photoEncoder.py | CLIP 模型变体3 |
| __init__.py | 模块初始化，导出主要模型类 |

### 2. 数据处理模块 astroclip/data/

| 文件 | 作用 |
|------|------|
| datamodule.py | **Lightning DataModule**，管理训练/验证/测试数据加载，协调分布式训练 |
| datamodule_photoEncoder.py | DataModule 变体1|
| datamodule_photoTransformer.py | PhotoTransformer 专用 DataModule 变体2|


### 3. 训练器模块 astroclip/

| 文件 | 作用 |
|------|------|
| trainer.py | **主训练器**，astroclip和specformer的训练脚本入口 |
| modules.py | **模型模块**，定义神经网络层和构建块 |
| callbacks.py | **训练回调**，包含模型检查点、日志、早停等回调函数 |
| scheduler.py | **学习率调度器**，实现各种学习率衰减策略 |
---

## 4. 图像预训练模块 astroclip/astrodino/

基于 DINOv2 的星系图像预训练模块。

### 数据增强子模块 astrodino/data/

| 文件 | 作用 |
|------|------|
| dataset.py | 自定义数据集 |
| augmentations.py | **核心增强策略**，数据增强操作 |

### 训练配置子模块 astrodino/

| 文件 | 作用 |
|------|------|
| trainer.py | **图像预训练训练器**，astrodino训练脚本入口 |
| config.yaml | **默认训练配置**，对应配置文件 |

---

## 5. 光度测量模块 astroclip/astrophoto/

测光编码器实现方式1


## 6. 照片编码器模块 astroclip/photoencoder/

独立的测光编码器模块，支持下游任务。（测光编码器实现方式2）

| 文件/目录 | 作用 |
|-----------|------|
| trainer.py | 测光编码器训练器 |
| mean_std_photo.py | 测光数据均值标准差 |

### 数据处理 photoencoder/data/

| 文件 | 作用 |
|------|------|
| dataset.py | 测光数据集 |
| datamodule.py | 数据模块 |
| photo_augmentations.py | 测光数据增强 |

### 模型定义 photoencoder/model/

| 文件 | 作用 |
|------|------|
| photoEncoder.py | **测光编码器模型** |

### 下游任务 photoencoder/downstream/

| 文件 | 作用 |
|------|------|
| models.py | 下游任务模型 |
| embed_photometry.py | 嵌入生成脚本 |
| evaluate_redshift.py | 红移评估 |

---

## configs/ 配置文件目录

| 文件 | 作用 |
|------|------|
| astroclip.yaml | **主 CLIP 模型训练配置**，astroclip配置文件 |
| specformer.yaml | **光谱 Transformer 训练配置**，光谱编码器specformer配置文件 |
| astroclip_photo.yaml | clip配置文件变体1 |
| astroclip_photoEncoder.yaml | clip配置文件变体2 |
| astroclip_ip.yaml | clip配置文件变体3 |
| astroclip_old.yaml | clip配置文件变体4 |
| astroclip.txt | 配置说明文本 |

---

## downstream_tasks/ 下游任务

用于验证预训练模型性能的基准任务。

| 文件 | 作用 |
|------|------|
| get_embedding_ip.py | 生成跨模态嵌入向量 |
| models.cpython-310.pyc | 编译后的模型模块 |

---

## dsm/ 数据分析与模拟

数据分析和光谱处理相关脚本。

| 文件/目录 | 作用 |
|-----------|------|
| analyze.ipynb | 分析星表 |
| visualize_spec_process.ipynb | 光谱处理可视化 |
| attention_map.ipynb | 注意力图可视化 |
| analyze_field_groups.py | 视场组分析 |
| build_astro_dataset_parallel_optim.py | 并行数据集构建 |
| sample_z15.py | 从data67W数据集中采样红移区间1-5的脚本 |
| csv/ | CSV 数据文件目录 |
| spect/png/ | 光谱 PNG 图像 |
| Gaussie | 高斯数据分析结果 |
| data_67W | 存放67W 数据集的均值和方差 |

---

## utils/ 工具脚本

| 文件/目录 | 作用 |
|-----------|------|
| analyze.md | 存放高斯噪声/模糊的分析结果 |
| analyze_parallel.py | 并行分析脚本 |
| mean_std_image.py | 图像数据均值标准差计算 |
| mean_std_photo.py | 照片数据均值标准差计算 |
| do.sh | 执行脚本 |
| photo | 存放测光均值标准差结果 |
| image | 存放图像均值标准差结果 |

---

## scripts/ 脚本目录

| 文件 | 作用 |
|------|------|
| patch_visualize.ipynb | 可视化vit的patch效果 |
| test_blur_noise.ipynb | 测试高斯模糊/噪声可视化效果 |
| sample.py | 采样小数据集，用于调试和测试 |

