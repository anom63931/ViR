#!/bin/bash
NUM_PROC=1
DATA_PATH="./ImageNet/ImageNet2012/validation"
checkpoint=./output/train/model_best.pth.tar
BS=128

python validate.py --model vir_small_patch16_224 --checkpoint=$checkpoint --data_dir=$DATA_PATH --batch-size $BS
