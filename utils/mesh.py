import argparse
import open3d as o3d
import numpy as np
import trimesh
import os
from fnmatch import fnmatch
import numpy as np
import multiprocessing as mp
import traceback
import mcubes
import trimesh
import torch
from skimage import measure
import sys
from tqdm import tqdm  # Import tqdm for progress bar
sys.path.append('/home/ubuntu/public_c/Sketch2Style')
from utils.other_utils import mesh_cut, smooth_border 

def process(udf, output_path):
    vertices, triangles = mcubes.marching_cubes(udf, 0.02)  # 0.0165
    vertices = (vertices.astype(np.float32) - 0.5) / 265 - 0.51

    mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(triangles))

    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=len(mesh.vertices)//5)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[cc[0]] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()

    final_mesh_cuple = mesh_cut(mesh, region_rate=20)
    final_mesh_1 = final_mesh_cuple[0]
    final_mesh_2 = final_mesh_cuple[1]

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
        v = np.array(final_mesh.vertices)
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

def process_directory(root_dir, error_log_file):
    with open(error_log_file, 'a') as log_file:
        # Collect all the files to be processed
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if fnmatch(filename, '*.npy'):
                    npy_file_path = os.path.join(dirpath, filename)
                    obj_file_name = filename.replace('.npy', '.obj')
                    obj_file_path = os.path.join(dirpath, obj_file_name)

                    # Skip if the corresponding .obj file already exists
                    if os.path.exists(obj_file_path):
                        print(f"Skipping already processed file: {npy_file_path}")
                        continue

                    # Load UDF data
                    if not os.path.exists(npy_file_path):
                        print(f"Error: UDF data file does not exist at {npy_file_path}", file=sys.stderr)
                        continue

                    file_paths.append((npy_file_path, obj_file_path))

        # Use tqdm to display progress bar
        with tqdm(total=len(file_paths), desc="Processing files", unit="file") as pbar:
            for npy_file_path, obj_file_path in file_paths:
                try:
                    udf_data = np.load(npy_file_path)
                    process(udf_data, obj_file_path)
                    print(f"Processed {npy_file_path} -> {obj_file_path}")
                except Exception as e:
                    # Log the error with the file name and error message
                    error_message = f"Error processing {npy_file_path}: {str(e)}\n{traceback.format_exc()}\n"
                    log_file.write(error_message)
                    print(f"Error processing {npy_file_path}. Logged to error log.")
                pbar.update(1)  # Update progress bar after each file is processed

def main():
    parser = argparse.ArgumentParser(description="Process UDF data and generate OBJ files.")
    parser.add_argument("--root_dir", type=str, default="./outputs/super", help="Root directory containing .npy files")
    parser.add_argument("--error_log", type=str, default="error_log_super.txt", help="Error log file path")
    args = parser.parse_args()

    process_directory(args.root_dir, args.error_log)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
