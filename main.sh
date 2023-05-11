#!/bin/bash

set -e

# set python path according to your actual environment
pythonpath='python'

# Training 
${pythonpath} main.py --dataset ED --woStrategy --batch_size 16 --pretrain --pretrain_epoch 4 --warmup 12000 --fine_weight 0.2 --coarse_weight 1.0 --seed 13 --gpu 1

# Testing
# ${pythonpath} main.py --dataset ED --test --gpu 1

# Evaluation
cp ./save/test/results.txt ./results/case.txt
${pythonpath} src/scripts/evaluate.py
