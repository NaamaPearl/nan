# NAN: Noise-Aware NeRFs for Burst Denoising

#### [project page](https://noise-aware-nerf.github.io) | [paper](https://arxiv.org/abs/2204.04668) 

[//]: # (| [model <mark>TODO]&#40;&#41;)
PyTorch implementation of the paper "NAN: Noise-Aware NeRFs for Burst Denoising", CVPR 2022.

> NAN: Noise-Aware NeRFs for Burst Denoising
> [Naama Pearl](mailto:naama.pearl@gmail.com) | [Tali Treibitz](https://www.viseaon.haifa.ac.il/) | [Simon Korman](https://www.cs.haifa.ac.il/~skorman/)
> CVPR, 2022


Our implementation is based on the paper "IBRNet: Learning Multi-View Image-Based Rendering" (CVPR 2021) and their [github repository](https://github.com/googleinterns/IBRNet).



## Installation
Clone this repository
```
git clone https://github.com/NaamaPearl/nan 
cd nan/
```

The code is tested with Python3.9, PyTorch==1.11 and cudatoolkit=11.1 on NIDIA RTX 3090. To create a conda environment compatible with RTX 3090:
```
conda env create -f environment.yml
conda activate nan
```
For different platforms, the pytorch installation will probably be different.

## Datasets
Please refer to [IBRNet](https://github.com/googleinterns/IBRNet) for the dataset instruction.

## Evaluation

Our checkpoints can be downloaded form [https://drive.google.com/file/d/1MFRdNA0Y9yowUEo991GvSjoUr8bYYglm/view](https://drive.google.com/file/d/1MFRdNA0Y9yowUEo991GvSjoUr8bYYglm/view),
or by using:
```
cd out
gdown https://drive.google.com/open?id=1MFRdNA0Y9yowUEo991GvSjoUr8bYYglm
unzip reproduce__NAN.zip

```

For evaluation run
```
python -m eval.evaluate
```
This will automatically load `eval.yml` and run evaluation for all scenes and all noise levels.
The checkpoint will be loaded from the path specified in `eval.yml`

[//]: # (## Rendering videos of smooth camera paths <mark>TODO)

[//]: # (Videos can be generated using)

[//]: # (```)

[//]: # (python -m eval.render_llff_video )

[//]: # (```)

## Training


This will train with nan configuration and load automatically `train.yml`
```
python train.py 
```
(IBRNet train with multiple GPUs. The framework for still exists, but we didn't run it.)
 
 
## Citation
```
 
@inproceedings{pearl2022noiseaware,
    title={NAN: Noise-Aware NeRFs for Burst-Denoising},
    author={Pearl, Naama and Treibitz, Tali and Korman, Simon},
    booktitle=CVPR,
    year={2022}
}

```
