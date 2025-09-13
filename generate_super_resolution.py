import numpy as np
import os
import argparse
import traceback
import mcubes
import trimesh
import open3d as o3d
from fnmatch import fnmatch
from tqdm import tqdm
import torch
from skimage import measure
from sparsity_network.sparsity_trainer import Sparsity_DiffusionModel
from utils.utils import str2bool, ensure_directory
from utils.sparsity_utils import load_input
from utils.mesh_utils import process_udf
from utils.other_utils import mesh_cut, smooth_border


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

    mesh = process_udf(res[0], level=level, normalize=True)
    np.save(os.path.join(output_path, f"{name}_udf.npy"), res[0])

    mesh.export(os.path.join(output_path, f"{name}.obj"))


def generate_meshes(
    model_path: str,
    npy_folder: str,
    batch_size: int,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 50,
    based_gt: bool = False,
    truncated_index: float = 0,
    level: float = 0.0,
    save_npy: bool = True,
    save_mesh: bool = True,
    start_index: int = 0,
    end_index: int = 100000,
):
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(model_path).cuda()

    save_path = os.path.join(output_path, "test")
    if based_gt:
        save_path += "_gt"
    ensure_directory(save_path)

    discrete_diffusion.generate_results_from_folder(
        folder=npy_folder, ema=ema,
        save_path=save_path, batch_size=batch_size, use_ddim=False, steps=steps,
        truncated_index=truncated_index, sort_npy=not based_gt, level=level,
        save_npy=save_npy, save_mesh=save_mesh, start_index=start_index, end_index=end_index
    )

    process_directory(save_path)


def process(udf, output_path):
    vertices, triangles = mcubes.marching_cubes(udf, 0.02)
    vertices = (vertices.astype(np.float32) - 0.5) / 265 - 0.51

    mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(triangles))

    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=len(mesh.vertices)//5)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[cc[0]] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()

    final_mesh_cuple = mesh_cut(mesh, region_rate=20)
    final_mesh_1, final_mesh_2 = final_mesh_cuple

    if len(final_mesh_1.vertices) > len(final_mesh_2.vertices):
        vertices = np.asarray(final_mesh_1.vertices)
        traingles = np.asarray(final_mesh_1.faces)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(traingles)
        mesh.compute_vertex_normals()
        mesh.normalize_normals()
        vertex_normals = np.array(mesh.vertex_normals)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        points = points + 0.01 * vertex_normals
        final_mesh = trimesh.Trimesh(points, faces)
    else:
        vertices = np.asarray(final_mesh_2.vertices)
        traingles = np.asarray(final_mesh_2.faces)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(traingles)
        mesh.compute_vertex_normals()
        mesh.normalize_normals()
        vertex_normals = np.array(mesh.vertex_normals)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        points = points - 0.01 * vertex_normals
        final_mesh = trimesh.Trimesh(points, faces)

    final_mesh = smooth_border(final_mesh)

    for i in range(3):
        points = np.array(final_mesh.vertices)
        faces = np.array(final_mesh.faces)
        index = trimesh.grouping.group_rows(final_mesh.edges_sorted, require_count=1)
        boundary_vertices = np.unique(final_mesh.edges[index].flatten())
        boundary_faces = []
        for i in range(faces.shape[0]):
            tmp = faces[i]
            for k in tmp:
                if k in boundary_vertices:
                    boundary_faces.append(i)
                    break
        faces = np.delete(faces, boundary_faces, axis=0)

        final_mesh = trimesh.Trimesh(points, faces)
        final_mesh.remove_unreferenced_vertices()

    final_mesh = trimesh.smoothing.filter_laplacian(final_mesh, iterations=12)
    for i in range(3):
        final_mesh = smooth_border(final_mesh)

    final_mesh.export(output_path)


def process_directory(root_dir):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch(filename, '*.npy'):
                npy_file_path = os.path.join(dirpath, filename)
                obj_file_name = filename.replace('.npy', '.obj')
                obj_file_path = os.path.join(dirpath, obj_file_name)

                if os.path.exists(obj_file_path):
                    print(f"Skipping already processed file: {npy_file_path}")
                    continue

                if not os.path.exists(npy_file_path):
                    print(f"Error: UDF data file does not exist at {npy_file_path}")
                    continue

                file_paths.append((npy_file_path, obj_file_path))

    with tqdm(total=len(file_paths), desc="Processing files", unit="file") as pbar:
        for npy_file_path, obj_file_path in file_paths:
            try:
                udf_data = np.load(npy_file_path)
                process(udf_data, obj_file_path)
                print(f"Processed {npy_file_path} -> {obj_file_path}")
            except Exception as e:
                print(f"Error processing {npy_file_path}: {str(e)}")
                print(traceback.format_exc())
            pbar.update(1)


if __name__ == '__main__':
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
