import pytorch_lightning as pl

from astroclip.photoencoder.data import PhotometryTransform, PhotometryDataset
from torch.utils.data import DataLoader


class PhotometryDataModule(pl.LightningDataModule):
    """
    负责数据加载的模块。
    """

    def __init__(self, data_path, batch_size=256, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 初始化预处理/增强模块
        # 注意：这里会使用你在 photo_augmentations.py 中填写的默认 Mean/Std
        # 如果你想在运行时传入 Mean/Std，可以在这里修改逻辑
        self.transform = PhotometryTransform()

    def setup(self, stage=None):
        # 加载训练集
        if stage == 'fit' or stage is None:
            self.train_dataset = PhotometryDataset(
                data_path=self.data_path,
                split='train',
                transform=self.transform
            )
            # 这里简单地把训练集的一部分划分为验证集，或者你可以加载单独的 test_dataset
            # 为了演示，假设我们有单独的 test 文件夹或者做简单的 split
            # 如果你有单独的验证集文件，请在这里修改
            self.val_dataset = PhotometryDataset(
                data_path=self.data_path,
                split='test',
                transform=self.transform
            )

            # 临时方案：划分 10% 做验证
            # full_size = len(self.train_dataset)
            # val_size = int(0.1 * full_size)
            # train_size = full_size - val_size
            # self.train_set, self.val_set = torch.utils.data.random_split(
            #     self.train_dataset, [train_size, val_size],
            #     generator=torch.Generator().manual_seed(42)
            # )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)
