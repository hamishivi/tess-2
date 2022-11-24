#!/bin/bash
# Training script to train the diffusion models. 

HF_HOME="/net/nfs.cirrascale/s2-research/rabeehk/.cache/huggingface/"
#HF_HOME=${HF_HOME} accelerate launch train.py $1

HF_HOME=${HF_HOME} accelerate launch --config_file configs/accelerate_4_gpus.yaml train.py  $1
# HF_HOME=${HF_HOME} accelerate launch --multi_gpu --mixed_precision "no" --num_processes 2 --num_machines 1  --num_cpu_threads_per_process 2  train.py  $1

