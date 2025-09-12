import copy
from tqdm import tqdm
from utils.utils import set_requires_grad
from utils.mesh_utils import process_udf
from torch.utils.data import DataLoader
from network.model_utils import EMA
from sparsity_network.data_loader import get_shapenet_sparsity_dataset, get_shapenet_sparsity_dataset_for_forward
from pathlib import Path
from torch.optim import AdamW
from utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from sparsity_network.sparsity_model import UDFDiffusion
import torch.nn as nn
from ocnn.octree import Octree
import os
import torch
import numpy as np
import ocnn
from utils.sparsity_utils import voxel2octree, octree2udfgrid


class Sparsity_DiffusionModel(LightningModule):
    def __init__(
        self,
        dataset_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        base_size: int = 32,
        upfactor: int = 2,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        ema_rate: float = 0.9999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        split_dataset: bool = False,
        data_augmentation: bool = False,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        udf_clip_value: float = 0.015,
        noise_schedule="linear",
        noise_level: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.results_folder = Path(results_folder)
        self.noise_level = noise_level
        self.model = UDFDiffusion(base_size=base_size,
                                  upfactor=upfactor,
                                  base_channels=base_channels,
                                  verbose=verbose,
                                  noise_schedule=noise_schedule,
                                  udf_clip_value=udf_clip_value)

        self.octree_feature = ocnn.modules.InputFeature('F', True)
        self.batch_size = batch_size
        self.lr = lr
        self.base_size = base_size
        self.upfactor = upfactor
        self.udf_clip_value = udf_clip_value
        self.image_size = base_size * upfactor
        self.dataset_folder = dataset_folder
        self.data_class = data_class
        self.save_every_epoch = save_every_epoch

        self.split_dataset = split_dataset
        self.data_augmentation = data_augmentation

        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.denoise_fn.parameters(), lr=self.lr)
        return [optimizer]

    #加载训练数据集
    def train_dataloader(self):
        dataset, collate_fn = get_shapenet_sparsity_dataset(self.dataset_folder, self.data_class,
                                                            size=self.image_size, udf_clip_value=self.udf_clip_value,
                                                            noise_level=self.noise_level,
                                                            split_dataset=self.split_dataset,
                                                            data_augmentation=self.data_augmentation)

        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            num_workers=os.cpu_count()//2,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False)

        return dataloader

    def get_input_feature(self, octree):
        data = self.octree_feature(octree)
        return data

    def training_step(self, batch, batch_idx):

        octree = batch['octree'] # 获取输入数据（octree）
        data = self.get_input_feature(octree) #将输入的 octree 数据转换为特征向量

        loss = self.model.training_loss(data, octree) # 计算损失

        loss = loss.mean()
        self.log("loss", loss.clone().detach().item(),
                 prog_bar=True, batch_size=self.batch_size)
        opt = self.optimizers() # 获取优化器

        opt.zero_grad()# 清除梯度
        self.manual_backward(loss) # 手动进行反向传播
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val) # 梯度裁剪
        opt.step()# 更新参数

        self.update_EMA() # 更新 EMA 模型

    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch)

        return super().on_train_epoch_end()

    @torch.no_grad()
    def generate_results_from_single_voxel(self, low_res_voxel, ema=True, steps=1000,
                                           use_ddim: bool = False, truncated_index: float = 0., verbose: bool = False): #输入：low_res_voxel低分辨率体素数据  输出：UDF 网格

        generator = self.ema_model if ema else self.model
        assert low_res_voxel.shape[-1] == self.base_size #断言输入的体素数据的尺寸与模型的预期大小一致
        octree = voxel2octree(low_res_voxel, upfactor=self.upfactor,
                              device=self.device) #低分辨率的体素数据转换为八叉树（Octree）结构

        data = self.octree_feature(octree) #提取八叉树的输入特征

        data = generator.sample(data, octree, use_ddim=use_ddim, steps=steps,
                                truncated_index=truncated_index, verbose=verbose) #使用扩散模型对数据进行采样（生成新的数据）。

        return octree2udfgrid(data, octree=octree, depth=octree.depth, scale=self.udf_clip_value, nempty=True) #将生成的 data 转换为一个 UDF 网格

    @torch.no_grad()
    def generate_results_from_folder(self, folder, save_path, ema=True, batch_size=8,
                                     steps=1000, use_ddim: bool = False, truncated_index: float = 0., sort_npy: bool = True, level: float = 0.0,
                                     save_npy: bool = True, save_mesh: bool = True, start_index: int = 0, end_index: int = 10000, verbose: bool = False):

        generator = self.ema_model if ema else self.model

        dataset, collate_fn = get_shapenet_sparsity_dataset_for_forward(
            folder, size=self.image_size, base_size=self.base_size, sort_npy=sort_npy, start_index=start_index, end_index=end_index)

        assert len(dataset) > 0
        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            num_workers=os.cpu_count()//2,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        index = start_index
        paths = dataset.get_paths()
        for _, batch in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Processing batches"):
            octree: Octree = batch['octree'].to(self.device)

            data = self.octree_feature(octree)
            data = generator.sample(
                data, octree, use_ddim=use_ddim, steps=steps, truncated_index=truncated_index, verbose=verbose)

            res = octree2udfgrid(data, octree=octree, depth=octree.depth,
                                scale=self.udf_clip_value, nempty=True)
            for i in range(octree.batch_size):
                field = res[i]
                original_path = paths[index]  # 获取原文件路径
                original_dir = os.path.dirname(original_path)  # 获取原文件所在目录
                original_filename = os.path.splitext(os.path.basename(original_path))[0]  # 获取原文件名（不带扩展名）
                relative_path = os.path.relpath(original_dir, folder)  # 相对路径
                save_dir = os.path.join(save_path, relative_path)  # 目标保存路径

                # 确保目标目录存在
                os.makedirs(save_dir, exist_ok=True)

                try:
                    if save_npy:
                        np.save(os.path.join(save_dir, f"{original_filename}.npy"), field)
                        print(os.path.join(save_dir, f"{original_filename}.npy"))
                    if save_mesh:
                        mesh = process_udf(field, level=level, normalize=True)
                        mesh.export(os.path.join(save_dir, f"{original_filename}.obj"))
                        print(os.path.join(save_dir, f"{original_filename}.obj"))
                except Exception as e:
                    print(str(e))
                index += 1
