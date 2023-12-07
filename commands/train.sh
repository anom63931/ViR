#!/bin/bash
NUM_PROC=1
DATA_PATH="./ImageNet/ImageNet2012/train"
MODEL=vir_small_patch16_224
BS=128
EXP=Test
LR=5e-3
WD=0.05
WR_LR=1e-6
DR=0.2

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=11223 train.py --input-size 3 224 224 \
--data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR
