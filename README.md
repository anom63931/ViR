# ViR: Towards Efficient Vision Retention Backbones

This repository is the official PyTorch implementation of **ViR: Towards Efficient Vision Retention Backbones**.

<p align="center">
<img src="https://github.com/anom63931/ViR/assets/118462623/4a8f50c6-39ac-46ec-bd31-6436efc12d01" width=63% height=63% 
class="center">
</p>

We propose a new class of computer vision models, dubbed Vision Retention Networks (ViR), with dual parallel and recurrent formulations, which strike an optimal balance between fast inference and parallel training with competitive performance. In particular, ViR scales favorably for image throughput and memory consumption in tasks that require higher-resolution images due to its flexible formulation in processing large sequence lengths. The ViR is the first attempt to realize dual parallel and recurrent equivalency in a general vision backbone for recognition tasks. We have validated the effectiveness of ViR through extensive experiments with different dataset sizes and various image resolutions and achieved competitive performance




## Installation

The dependencies can be installed by running:

```bash
pip install -r requirements.txt
```


## Training

The `ViR` model can be trained from scratch on ImageNet-1K dataset by running:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus> --master_port 11223  train.py \ 
--model <model-name> --data_dir <imagenet-path> --batch-size --amp <batch-size-per-gpu> --tag <run-tag> --model-ema
```

To resume training from a pre-trained checkpoint:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus> --master_port 11223  train.py \ 
--resume <checkpoint-path> --model <model-name> --amp --data_dir <imagenet-path> --batch-size <batch-size-per-gpu> --tag <run-tag> --model-ema
```

## ImageNet-1K Data Preparation

Please download the ImageNet dataset from its official website. The training and validation images need to have
sub-folders for each class with the following structure:

```bash
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

## Licenses

For license information regarding the timm repository, please refer to the [official website](https://github.com/rwightman/pytorch-image-models).

For license information regarding the ImageNet dataset, please refer to the [official website](https://www.image-net.org/). 
