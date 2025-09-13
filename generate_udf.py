import torch
import numpy as np
from network.model_trainer import DiffusionModel
from utils.mesh_utils import voxel2mesh
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups
import argparse
import os
from tqdm import tqdm
from utils.utils import VIT_MODEL, png_fill_color
from PIL import Image
import timm
from utils.sketch_utils import _transform, create_random_pose, get_P_from_transform_matrix


def generate_based_on_sketch_batch(
    model_path: str,
    sketch_file: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 1,
    steps: int = 50,
    truncated_time: float = 0.0,
    w: float = 1.0,
    view_information: int = 0,
    kernel_size: float = 2,
    detail_view: bool = False,
    rotation: float = 0.0,
    elevation: float = 0.0,
):
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()

    # Read the sketch file, each line contains a subfolder name
    with open(sketch_file, "r") as f:
        subfolders = f.readlines()

    # Set up the root directory for output
    root_dir = os.path.join(output_path, f"{model_name}_{model_id}_batch_{ema}")
    ensure_directory(root_dir)
    preprocess = _transform(224)
    device = "cuda"

    feature_extractor = timm.create_model(VIT_MODEL, pretrained=True).to(device)

    for subfolder in subfolders:
        subfolder = subfolder.strip()  # Clean any extra whitespace or newline characters
        subfolder_path = os.path.join("/home/ubuntu/public_d/cxy/DeepSketch2Wear/datasets/sketch", subfolder)

        # Check if the subfolder exists and the image is present
        sketch_path = os.path.join(subfolder_path, "edge_front.png")
        if not os.path.exists(sketch_path):
            print(f"Skipping {sketch_path}, file does not exist.")
            continue

        image_name = subfolder  # Use subfolder name as the image name
        postfix = f"{model_name}_{model_id}_{ema}_{image_name}_{w}_{view_information}"

        # Create a subdirectory for each sketch inside the root directory
        sketch_output_dir = os.path.join(root_dir, image_name)
        ensure_directory(sketch_output_dir)

        # Reset index for each subfolder
        index = 0

        with torch.no_grad():
            im = Image.open(sketch_path)
            im = png_fill_color(im).convert("RGB")
            im.save(os.path.join(sketch_output_dir, "input.png"))
            im = preprocess(im).unsqueeze(0).to(device)
            image_features = feature_extractor.forward_features(im)
            sketch_c = image_features.squeeze(0).cpu().numpy()

        # Handle projection matrix for view
        from utils.sketch_utils import Projection_List, Projection_List_zero
        if detail_view:
            projection_matrix = get_P_from_transform_matrix(create_random_pose(rotation=rotation, elevation=elevation))
        elif view_information == -1:
            projection_matrix = None
        else:
            if discrete_diffusion.elevation_zero:
                projection_matrix = Projection_List_zero[view_information]
            else:
                projection_matrix = Projection_List[view_information]

        # Split into batches for generation
        batches = num_to_groups(num_generate, 32)
        generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model

        # Generate 3D objects for each batch
        for batch in batches:
            res_tensor = generator.sample_with_sketch(
                sketch_c=sketch_c, batch_size=batch,
                projection_matrix=projection_matrix, kernel_size=kernel_size,
                steps=steps, truncated_index=truncated_time, sketch_w=w
            )
            for i in tqdm(range(batch), desc=f'Saving results for {image_name} in {sketch_output_dir}'):
                voxel = res_tensor[i].squeeze().cpu().numpy()
                np.save(os.path.join(sketch_output_dir, f"{index}.npy"), voxel)

                try:
                    voxel[voxel > 0] = 1
                    voxel[voxel < 0] = 0
                    mesh = voxel2mesh(voxel)
                    mesh.export(os.path.join(sketch_output_dir, f"{index}.obj"))
                except Exception as e:
                    print(f"Error processing voxel for {image_name} at index {index}: {str(e)}")

                # Increment index after processing each voxel
                index += 1



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Batch generate based on sketches')
    parser.add_argument("--generate_method", type=str, default='generate_based_on_sketch_batch',
                        help="Choose method: 'generate_based_on_sketch_batch'")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--sketch_file", type=str, required=True, help="Text file containing a list of subfolder names")
    parser.add_argument("--text_w", type=float, default=1.0)
    parser.add_argument("--image_name", type=str2bool, default=False)
    parser.add_argument("--sketch_w", type=float, default=1.0)
    parser.add_argument("--view_information", type=int, default=0)
    parser.add_argument("--detail_view", type=str2bool, default=False)
    parser.add_argument("--rotation", type=float, default=0.)
    parser.add_argument("--elevation", type=float, default=0.)
    parser.add_argument("--kernel_size", type=float, default=4.)
    parser.add_argument("--verbose", type=str2bool, default=False)

    args = parser.parse_args()

    method = args.generate_method.lower()
    ensure_directory(args.output_path)

    if method == "generate_based_on_sketch_batch":
        generate_based_on_sketch_batch(
            model_path=args.model_path,
            sketch_file=args.sketch_file,
            output_path=args.output_path,
            ema=args.ema,
            num_generate=args.num_generate,
            steps=args.steps,
            truncated_time=args.truncated_time,
            w=args.sketch_w,
            view_information=args.view_information,
            kernel_size=args.kernel_size,
            detail_view=args.detail_view,
            rotation=args.rotation,
            elevation=args.elevation
        )
    else:
        raise NotImplementedError("Method not implemented.")
