# NAN: Noise-Aware NeRFs for Burst Denoising

#### [project page <mark>TODO](https://noise-aware-nerf.github.io) | [paper](https://arxiv.org/abs/2204.04668) | [data & model <mark>TODO]()
PyTorch implementation of the paper "NAN: Noise-Aware NeRFs for Burst Denoising", CVPR 2022.

> NAN: Noise-Aware NeRFs for Burst Denoising
> [Naama Pearl](mailto:naama.pearl@gmail.com) | [Tali Treibitz]() | [Simon Korman]()
> CVPR, 2022


Our implementation is based on the paper "IBRNet: Learning Multi-View Image-Based Rendering" (CVPR 2021) and their [github repository](https://github.com/googleinterns/IBRNet).



## Installation
Clone this repository <mark>TODO
```
git clone --recurse-submodules https://github.com/NaamaPearl/nan 
cd nan/
```

The code is tested with Python3.8, PyTorch == *** and CUDA == ***. To create the anaconda environment: <mark>TODO
```
conda env create -f environment.yml
conda activate nan
```

## Datasets
Please refer to [IBRNet](https://github.com/googleinterns/IBRNet) for the dataset instruction.

## Evaluation
Our checkpoints can be downloaded using: <mark>TODO
```
gdown ****
unzip pretrained_model.zip
```

For evaluation run <mark>TODO
```
python -m eval.eval_wrraper
```
This will automatically load `eval.yml` and run evaluation for all scenes and all noise levels.
The checkpoint will be loaded from the path specified in `eval.yml`

## Rendering videos of smooth camera paths <mark>TODO
Videos can be geenerated using
```
python -m eval.render_llff_video 
```

## Training

```
This will train with nan configuration and load automatically `train.yml`
```
python train.py 
```
(IBRNet train with multiple GPUs. The framework for still exists, but I didn't test it.)
 
 
## Citation <mark>TODO
```
 
@inproceedings{****
}

```
