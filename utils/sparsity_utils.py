from ocnn.octree import Points, Octree, key2xyz
import numpy as np
import torch
from ocnn.nn.octree_pad import octree_depad
from scipy.ndimage import binary_fill_holes


def generate_template(factor):
    template = np.zeros((factor ** 3, 3), dtype=np.int16)
    for i in range(factor):
        for j in range(factor):
            for k in range(factor):
                template[i*factor**2 + j*factor + k, 0] = i
                template[i*factor**2 + j*factor + k, 1] = j
                template[i*factor**2 + j*factor + k, 2] = k

    return template

#体素（voxel）转换为索引
def voxel2indices(voxel, depadding=False):
    assert type(voxel) == np.ndarray
    indices = np.stack(np.nonzero(voxel)).transpose()
    if depadding:
        template = generate_template(2)
        low_indices = np.unique(indices//2, axis=0)
        indices = np.repeat(low_indices, 8, axis=0) * 2 + \
            np.repeat(template[None, :], len(
                low_indices), axis=0).reshape(-1, 3)
    return indices


#根据voxel2indices的索引，将体素数据转换为八叉树结构
def voxel2octree(voxel, upfactor=2, device="cuda:0"):

    return indices2octree(indices=voxel2indices(voxel),
                          res=voxel.shape[-1],
                          upfactor=upfactor,
                          device=device)

#将索引转换为八叉树（octree）
def indices2octree(indices, res=32, upfactor=2, device="cuda:0"):
    template = generate_template(upfactor)

    indices = np.repeat(indices, upfactor ** 3, axis=0) * upfactor + \
        np.repeat(template[None, :], len(indices), axis=0).reshape(-1, 3)
    res *= upfactor

    xyz = (2 * indices + 1) / res - 1
    features = torch.ones((len(xyz), 1))

    points = Points(torch.from_numpy(xyz), features=features).to(device=device)
    points.clip(min=-1, max=1)

    depth = int(np.log2(res))
    octree = Octree(depth, 1, device=device)
    octree.build_octree(points)
    octree.construct_all_neigh()

    return octree


#将八叉树中的数据转换为 SDF 网格
def octree2sdfgrid(data: torch.Tensor, octree: Octree, depth: int, scale: float,   #scale=0.015
                   nempty: bool = False):
    #获取当前层次的Octree键（key）
    key = octree.keys[depth]
    if nempty:
        key = octree_depad(key, octree, depth)

    #将Octree键转换为坐标
    x, y, z, b = key2xyz(key, depth)
    num = 1 << depth  # 计算每个轴上的体素数目
    batch_size = octree.batch_size  # 获取批量大小
    size = (batch_size, num, num, num)  # 体素网格的形状
    vox = torch.ones(size, dtype=data.dtype, device=data.device) * scale  # 初始化体素网格，所有值为scale
    mask = torch.zeros(size, dtype=torch.long, device=data.device)  # 初始化掩码

    # vox[b, x, y, z] = torch.abs(data[:, 0]) * scale
    # mask[b, x, y, z] = torch.ones_like(data[:, 0]).to(torch.long)
    # vox = vox.cpu().numpy()
    # mask = mask.cpu().numpy()

    # for _batch in range(batch_size):
    #     _mask = binary_fill_holes(mask[_batch]).astype(np.int32) - mask[_batch]
    #     x, y, z = np.nonzero(_mask)
    #     vox[_batch, x, y, z] = np.abs(scale)

    # 将八叉树节点中存储的数据赋值到SDF网格中
    vox[b, x, y, z] = data[:, 0] * scale #将输入数据映射到体素网格
    mask[b, x, y, z] = torch.ones_like(data[:, 0]).to(torch.long)# 标记已经填充的体素
    vox = vox.cpu().numpy()
    mask = mask.cpu().numpy()
    
    #处理体素网格中未被直接从八叉树节点数据填充的部分
    for _batch in range(batch_size):
        #空洞的处理
        _mask = binary_fill_holes(mask[_batch]).astype(np.int32) - mask[_batch]# 填充空洞
        x, y, z = np.nonzero(_mask)# 获取空洞的位置
        vox[_batch, x, y, z] = -scale# 将这些位置的SDF值设置为负值

    return vox


def load_input(npy_path):
    low_res_voxel = np.load(npy_path)
    low_res_voxel[low_res_voxel > 0] = 1 #将所有大于 0 的值设置为 1，表示该体素为“有物体”
    low_res_voxel[low_res_voxel < 0] = 0 #将所有小于 0 的值设置为 0，表示该体素为“无物体”
    return low_res_voxel #二值化的体素数组
