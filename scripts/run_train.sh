#!/bin/bash
# Training script to train the diffusion models. 
# Usage: bash scripts/run_train.sh  ACCELERATE_CONFIG  SCRIPT_CONFIG
# Example: bash scripts/run_train.sh  configs/accelerate_1_gpu.yaml  configs/simple_data_test.json

HF_HOME="/net/nfs.cirrascale/s2-research/rabeehk/.cache/huggingface/"
#HF_HOME="/net/nfs/allennlp/rabeehk/.cache/huggingface/"
HF_HOME=${HF_HOME} accelerate launch --config_file $1 train.py  $2
# HF_HOME=${HF_HOME} accelerate launch --multi_gpu --mixed_precision "no" --num_processes 2 --num_machines 1  --num_cpu_threads_per_process 2  train.py  $1

