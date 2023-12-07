# FasterViT: Fast Vision Transformers with Hierarchical Attention

This repository is the official PyTorch implementation of **FasterViT: Fast Vision Transformers with Hierarchical Attention**.

<p align="center">
<img src="./assets/plot.png" width=62% height=62% 
class="center">
</p>

We introduce FasterViT which achieves a SOTA Pareto-front in
terms of accuracy vs. image throughput trade-off. Specifically, we propose
a new self-attention mechanism, denoted as Hierarchical
Attention, that captures long-range information by learning
cross-window carrier tokens and that scales linearly with
image resolution. We have extensively validated the effectiveness of FasterViT on various computer vision tasks including
ImageNet-1K for classification, MS COCO for object detection and instance segmentation, and ADE20K for semantic
segmentation.


The Hierarchical Attention is demonstrated in the following. Carrier tokens learn a summary of each region and interact with them to model
the cross-window global dependencies:

![teaser](./assets/hierarchial_attn.png)


## Results on ImageNet

### ImageNet-1K
**FasterViT ImageNet-1K Pretrained Models**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Throughput(Img/Sec)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
  </tr>

<tr>
    <td>FasterViT-0</td>
    <td>82.1</td>
    <td>95.9</td>
    <td>5802</td>
    <td>224x224</td>
    <td>31.4</td>
    <td>3.3</td>
</tr>

<tr>
    <td>FasterViT-1</td>
    <td>83.2</td>
    <td>96.5</td>
    <td>4188</td>
    <td>224x224</td>
    <td>53.4</td>
    <td>5.3</td>
</tr>

<tr>
    <td>FasterViT-2</td>
    <td>84.2</td>
    <td>96.8</td>
    <td>3161</td>
    <td>224x224</td>
    <td>75.9</td>
    <td>8.7</td>
</tr>

<tr>
    <td>FasterViT-3</td>
    <td>84.9</td>
    <td>97.2</td>
    <td>1780</td>
    <td>224x224</td>
    <td>159.5</td>
    <td>18.2</td>
</tr>

<tr>
    <td>FasterViT-4</td>
    <td>85.4</td>
    <td>97.3</td>
    <td>849</td>
    <td>224x224</td>
    <td>424.6</td>
    <td>36.6</td>
</tr>

<tr>
    <td>FasterViT-5</td>
    <td>85.6</td>
    <td>97.4</td>
    <td>449</td>
    <td>224x224</td>
    <td>975.5</td>
    <td>113.0</td>
</tr>

<tr>
    <td>FasterViT-6</td>
    <td>85.8</td>
    <td>97.4</td>
    <td>352</td>
    <td>224x224</td>
    <td>1360.0</td>
    <td>142.0</td>
</tr>

</table>

### ImageNet-21K
**FasterViT ImageNet-21K Pretrained Models (ImageNet-1K Fine-tuned)**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
  </tr>

<tr>
    <td>FasterViT-4-21K-224</td>
    <td>86.6</td>
    <td>97.8</td>
    <td>224x224</td>
    <td>271.9</td>
    <td>40.8</td>
</tr>

<tr>
    <td>FasterViT-4-21K-384</td>
    <td>87.6</td>
    <td>98.3</td>
    <td>384x384</td>
    <td>271.9</td>
    <td>120.1</td>
</tr>

<tr>
    <td>FasterViT-4-21K-512</td>
    <td>87.8</td>
    <td>98.4</td>
    <td>512x512</td>
    <td>271.9</td>
    <td>213.5</td>
</tr>

<tr>
    <td>FasterViT-4-21K-768</td>
    <td>87.9</td>
    <td>98.5</td>
    <td>768x768</td>
    <td>271.9</td>
    <td>480.4</td>
</tr>

</table>


### Robustness (ImageNet-A - ImageNet-R - ImageNet-V2)


All models use `crop_pct=0.875`. Results are obtained by running inference on ImageNet-1K pretrained models without finetuning.
<table>
  <tr>
    <th>Name</th>
    <th>A-Acc@1(%)</th>
    <th>A-Acc@5(%)</th>
    <th>R-Acc@1(%)</th>
    <th>R-Acc@5(%)</th>
    <th>V2-Acc@1(%)</th>
    <th>V2-Acc@5(%)</th>
  </tr>

<tr>
    <td>FasterViT-0</td>
    <td>23.9</td>
    <td>57.6</td>
    <td>45.9</td>
    <td>60.4</td>
    <td>70.9</td>
    <td>90.0</td>
</tr>

<tr>
    <td>FasterViT-1</td>
    <td>31.2</td>
    <td>63.3</td>
    <td>47.5</td>
    <td>61.9</td>
    <td>72.6</td>
    <td>91.0</td>
</tr>

<tr>
    <td>FasterViT-2</td>
    <td>38.2</td>
    <td>68.9</td>
    <td>49.6</td>
    <td>63.4</td>
    <td>73.7</td>
    <td>91.6</td>
</tr>

<tr>
    <td>FasterViT-3</td>
    <td>44.2</td>
    <td>73.0</td>
    <td>51.9</td>
    <td>65.6</td>
    <td>75.0</td>
    <td>92.2</td>
</tr>

<tr>
    <td>FasterViT-4</td>
    <td>49.0</td>
    <td>75.4</td>
    <td>56.0</td>
    <td>69.6</td>
    <td>75.7</td>
    <td>92.7</td>
</tr>

<tr>
    <td>FasterViT-5</td>
    <td>52.7</td>
    <td>77.6</td>
    <td>56.9</td>
    <td>70.0</td>
    <td>76.0</td>
    <td>93.0</td>
</tr>

<tr>
    <td>FasterViT-6</td>
    <td>53.7</td>
    <td>78.4</td>
    <td>57.1</td>
    <td>70.1</td>
    <td>76.1</td>
    <td>93.0</td>
</tr>

</table>

A, R and V2 denote ImageNet-A, ImageNet-R and ImageNet-V2 respectively. 

## Installation

The dependencies can be installed by running:

```bash
pip install -r requirements.txt
```

### Evaluation

To evaluate a pre-trained checkpoint using ImageNet-1K validation set:

```bash
python validate.py --model <model-name> --checkpoint <checkpoint-path> --data_dir <imagenet-path> --batch-size <batch-size-per-gpu>
```

## Training

The `Faster ViT` model can be trained from scratch on ImageNet-1K dataset by running:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus> --master_port 11223  train.py \ 
--config <config-file> --data_dir <imagenet-path> --batch-size --amp <batch-size-per-gpu> --tag <run-tag> --model-ema
```

To resume training from a pre-trained checkpoint:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus> --master_port 11223  train.py \ 
--resume <checkpoint-path> --config <config-file> --amp --data_dir <imagenet-path> --batch-size <batch-size-per-gpu> --tag <run-tag> --model-ema
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
