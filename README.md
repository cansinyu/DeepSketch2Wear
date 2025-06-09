<p align="center">
<h1 align="center">DeepSketch2Wear: Democratizing 3D Garment Creation via Freehand Sketches and Text</h1>

## Installation

We recommend creating an [anaconda](https://www.anaconda.com/) environment using our provided `environment.yml`:

```
conda env create -f environment.yml
conda activate sketch2style
```

## Data Preparation
Please download the required UDF data and sketch data [here](https://pan.baidu.com/)(password: 6khe). 
### UDF data creation
Please ref to [UDiFF](https://github.com/weiqi-zhang/UDiFF/tree/main) for generating the UDF field from Deepfashion data or your customized data.
### Sketch data creation
Please refer to  `prepare_sketch.py` for details.

## Training & Inference
### Stage 1: Base Diffusion Model
Training:
```
bash scripts/train.sh
```
Inference:
```
bash scripts/generate.sh
```
### Stage 2: Subdivision Diffusion Model
Training:
```
bash scripts/train_super.sh
```
Inference:
```
bash scripts/generate_super.sh
```
Note: All scripts are located in the `scripts/` directory. Refer to the individual script files for additional configuration options and parameters.

## Texture Generation

For texture generation, we recommend using [SyncMVD](https://github.com/LIU-Yuxin/SyncMVD). Follow these steps:

### 1. Clone the SyncMVD Repository:
```
git clone https://github.com/LIU-Yuxin/SyncMVD.git
cd SyncMVD
```
### 2. Follow SyncMVD's Setup Instructions to install dependencies and prepare the environment.
### 3. Generate Textures:
Use SyncMVD with the output meshes from our model (saved in ./output/) as input.
