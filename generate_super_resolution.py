import numpy as np
from sparsity_network.sparsity_trainer import Sparsity_DiffusionModel
from utils.utils import str2bool, ensure_directory
from utils.sparsity_utils import load_input
from utils.mesh_utils import process_sdf
import argparse
import os


def generate_one_mesh(
    model_path: str,
    npy_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 1000,
    truncated_index: float = 0,
    level: float = 0.0,
):
    npy_name = npy_path.split('/')[-1].split(".")[0]
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(
        model_path).cuda()

    res = discrete_diffusion.generate_results_from_single_voxel(
        low_res_voxel=load_input(npy_path), ema=ema, use_ddim=False, steps=steps, truncated_index=truncated_index)

    name = f"{model_name}_{model_id}_{ema}_ddpm_{steps}_{npy_name}_{truncated_index}"

    mesh = process_sdf(res[0], level=level, normalize=True)
    np.save(os.path.join(output_path, f"{name}_sdf.npy"), res[0])

    mesh.export(os.path.join(output_path, f"{name}.obj"))


def generate_meshes(
    model_path: str,
    npy_folder: str,
    batch_size: int,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 50, #1000
    based_gt: bool = False,
    truncated_index: float = 0,
    level: float = 0.0,
    save_npy: bool = True,
    save_mesh: bool = True,
    start_index: int = 0,
    end_index: int = 100000,
):
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(model_path).cuda()

    save_path = os.path.join(output_path, "text")

    if based_gt:
        save_path += "_gt"
    ensure_directory(save_path)

    # 遍历npy_folder下的所有一级子文件夹
    # first_level_subfolders = [f.path for f in os.scandir(npy_folder) if f.is_dir()]

    # selected_npy_files = []

    # # 遍历npy_folder下的所有一级子文件夹
    # for first_level_subfolder in os.listdir(npy_folder):
    #     first_level_path = os.path.join(npy_folder, first_level_subfolder)
    #     if os.path.isdir(first_level_path) and first_level_subfolder != 'super':
    #         second_level_subfolders = [f.path for f in os.scandir(first_level_path) if f.is_dir()]
        
    #         # 如果二级子文件夹数量小于3，则选择所有二级子文件夹
    #         if len(second_level_subfolders) <= 3:
    #             selected_second_level_subfolders = second_level_subfolders
    #         else:
    #             # 否则，随机选择3个二级子文件夹
    #             selected_second_level_subfolders = random.sample(second_level_subfolders, 3)
              
    #         # 遍历每个选中的二级子文件夹
    #         for second_level_subfolder in selected_second_level_subfolders:
    #             # 获取当前二级子文件夹下的所有npy文件路径
    #             npy_files = glob.glob(os.path.join(second_level_subfolder, '*.npy'))
    #             selected_npy_files.extend(npy_files)        

    # npy_files = glob.glob(os.path.join(npy_folder, '**/*.npy'), recursive=True) #all
    # print("Found .npy files:", npy_files)
  
    # for npy_file in tqdm(npy_files, desc="Generating Meshes"):
    #     relative_path = os.path.relpath(npy_file, npy_folder)
    #     result_folder = os.path.join(save_path, os.path.dirname(relative_path))
    #     ensure_directory(result_folder)
        
    discrete_diffusion.generate_results_from_folder(
        folder=npy_folder, ema=ema,
        save_path=save_path, batch_size=batch_size, use_ddim=False, steps=steps,
        truncated_index=truncated_index, sort_npy=not based_gt, level=level,
        save_npy=save_npy, save_mesh=save_mesh, start_index=start_index, end_index=end_index)
    

        # generate_one_mesh(
        #     model=discrete_diffusion, 
        #     npy_path=npy_file, 
        #     output_path=result_folder, 
        #     ema=ema, 
        #     steps=steps, 
        #     truncated_index=truncated_index, 
        #     level=level,
        #     save_npy=save_npy,  # 确保传递这些参数
        #     save_mesh=save_mesh
        # )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='generate_one_mesh',
                        help="please choose :\n \
                       1. 'generate_one_mesh'\n \
                       2. 'generate_meshes' \
                       ")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--npy_path", type=str, default="./test.npy")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--based_gt", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_npy", type=str2bool, default=True)
    parser.add_argument("--save_mesh", type=str2bool, default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--level", type=float, default=0.0)

    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)
    if method == "generate_one_mesh":
        generate_one_mesh(model_path=args.model_path, npy_path=args.npy_path, output_path=args.output_path, steps=args.steps, level=args.level,
                          ema=args.ema, truncated_index=args.truncated_time)
    elif method == "generate_meshes":
        generate_meshes(model_path=args.model_path, npy_folder=args.npy_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                        batch_size=args.batch_size, based_gt=args.based_gt, truncated_index=args.truncated_time, level=args.level, save_npy=args.save_npy, save_mesh=args.save_mesh,
                        start_index=args.start_index, end_index=args.end_index)
    else:
        raise NotImplementedError
