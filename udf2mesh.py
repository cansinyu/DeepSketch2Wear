import argparse
import open3d as o3d
import numpy as np
import trimesh
import os
from fnmatch import fnmatch
import mcubes
import torch
from skimage import measure
import sys
from tqdm import tqdm
sys.path.append('/home/ubuntu/public_c/DeepSketch2Wear')
from utils.seg_utils import mesh_cut, smooth_border

def process(udf, output_path):
    # 1. Convert the UDF to a mesh using marching cubes
    vertices, triangles = mcubes.marching_cubes(udf, 0.02)
    vertices = (vertices.astype(np.float32) - 0.5) / 265 - 0.51
    mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(triangles))

    # 2. Keep largest connected component
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=len(mesh.vertices)//5)
    if len(cc) == 0:
        print(f"Warning: mesh is empty after connected component filtering, skipping {output_path}")
        return
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[cc[0]] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()

    # 3. Graph cut segmentation
    final_mesh_cuple = mesh_cut(mesh, region_rate=20)
    if final_mesh_cuple is None:
        print(f"Warning: mesh_cut failed, skipping {output_path}")
        return

    final_mesh_1, final_mesh_2 = final_mesh_cuple
    if len(final_mesh_1.vertices) > len(final_mesh_2.vertices):
        vertices, faces = np.asarray(final_mesh_1.vertices), np.asarray(final_mesh_1.faces)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.normalize_normals()
        points = np.array(mesh.vertices) + 0.01 * np.array(mesh.vertex_normals)
        final_mesh = trimesh.Trimesh(points, faces)
    else:
        vertices, faces = np.asarray(final_mesh_2.vertices), np.asarray(final_mesh_2.faces)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.normalize_normals()
        points = np.array(mesh.vertices) - 0.01 * np.array(mesh.vertex_normals)
        final_mesh = trimesh.Trimesh(points, faces)

    # 4. Smooth border
    final_mesh = smooth_border(final_mesh)

    for _ in range(3):
        points, faces = np.array(final_mesh.vertices), np.array(final_mesh.faces)
        index = trimesh.grouping.group_rows(final_mesh.edges_sorted, require_count=1)
        boundary_vertices = np.unique(final_mesh.edges[index].flatten())
        boundary_faces = [i for i, tri in enumerate(faces) if any(v in boundary_vertices for v in tri)]
        faces = np.delete(faces, boundary_faces, axis=0)
        final_mesh = trimesh.Trimesh(points, faces)
        final_mesh.remove_unreferenced_vertices()

    final_mesh = trimesh.smoothing.filter_laplacian(final_mesh, iterations=12)
    for _ in range(3):
        final_mesh = smooth_border(final_mesh)

    final_mesh.export(output_path)
    print(f"Processed and saved: {output_path}")

def process_directory(root_dir):
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch(filename, '*.npy'):
                npy_file_path = os.path.join(dirpath, filename)
                obj_file_name = filename.replace('.npy', '.obj')
                obj_file_path = os.path.join(dirpath, obj_file_name)

                if os.path.exists(obj_file_path):
                    print(f"Skipping already processed file: {npy_file_path}")
                    continue
                if not os.path.exists(npy_file_path):
                    print(f"Error: UDF data file does not exist at {npy_file_path}", file=sys.stderr)
                    continue
                file_paths.append((npy_file_path, obj_file_path))

    with tqdm(total=len(file_paths), desc="Processing files", unit="file") as pbar:
        for npy_file_path, obj_file_path in file_paths:
            try:
                udf_data = np.load(npy_file_path)
                process(udf_data, obj_file_path)
            except Exception as e:
                print(f"Error processing {npy_file_path}: {e}")
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Process UDF data and generate OBJ files.")
    parser.add_argument("--root_dir", type=str, default="./outputs/4-13", help="Root directory containing .npy files")
    args = parser.parse_args()
    process_directory(args.root_dir)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
