#!/bin/bash
# Training script to train the diffusion models. 

accelerate launch train.py $1
