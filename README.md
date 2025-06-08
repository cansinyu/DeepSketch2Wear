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
### SDF data creation
Please ref to [UDiFF](https://github.com/weiqi-zhang/UDiFF/tree/main) for generating the UDF field from Deepfashion data or your customized data.
### Sketch data creation
Please refer to  `prepare_sketch.py` for details.

## Training
### Stage 1: Base Diffusion Model
To train the base diffusion model:
```
bash scripts/train.sh

```
### Stage 2: Subdivision Diffusion Model
To train the enhanced subdivision model:
```
bash scripts/train_super.sh

```
Note: All training scripts are located in the `scripts/` directory. Refer to the individual script files for additional configuration options and parameters.
