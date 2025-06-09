import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import timm
import fire
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDPPlugin

from utils.sketch_utils import _transform
from utils.utils import VIT_MODEL, set_requires_grad, ensure_directory


class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, sketch_folder: str, out_dir: str,
                 image_resolution: int = 224,
                 start_index: int = 0, end_index: int = 10000):
        super().__init__()
        self.preprocess = _transform(image_resolution)
        self.out_dir = out_dir
        self.sketch_paths = sorted(list(Path(sketch_folder).glob('*.png')))[start_index:end_index]
        
        # Create output directory
        ensure_directory(os.path.join(out_dir, "feature"))

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        model_name = sketch_path.stem
        image = Image.open(sketch_path).convert("RGB")
        image_tensor = self.preprocess(image)
        return {'image': image_tensor, 'model_name': model_name}


class SketchFeatureExtractor(LightningModule):
    def __init__(self,
                 sketch_folder: str,
                 results_folder: str = './results',
                 start_index: int = 0,
                 end_index: int = 100000):
        super().__init__()
        self.num_workers = os.cpu_count()
        self.sketch_folder = sketch_folder
        self.out_dir = results_folder
        self.start_index = start_index
        self.end_index = end_index

        # Initialize ViT feature extractor
        self.feature_extractor = timm.create_model(VIT_MODEL, pretrained=True)
        set_requires_grad(self.feature_extractor, False)

    def train_dataloader(self):
        dataset = SketchDataset(
            sketch_folder=self.sketch_folder,
            out_dir=self.out_dir,
            start_index=self.start_index,
            end_index=self.end_index
        )
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

    def training_step(self, batch, batch_idx):
        image = batch['image']
        model_name = batch['model_name'][0]

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor.forward_features(image).squeeze().cpu().numpy()

        # Save features to .npy
        out_path = os.path.join(self.out_dir, "feature", f"{model_name}.npy")
        np.save(out_path, features)

        return None  # No loss needed

    def configure_optimizers(self):
        return None  # No optimization needed


def extract_sketch_features(
    sketch_folder: str = "/home/ubuntu/public_c/cxy/data/las_udf/udf",
    results_folder: str = "/home/ubuntu/public_c/cxy/data/las_udf/sketchs",
    start_index: int = 0,
    end_index: int = 100000
):
    """Main function: extract features from sketch images"""
    model = SketchFeatureExtractor(
        sketch_folder=sketch_folder,
        results_folder=results_folder,
        start_index=start_index,
        end_index=end_index
    )

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        max_epochs=1
    )

    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(extract_sketch_features)
