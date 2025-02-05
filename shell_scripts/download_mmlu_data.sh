#!/bin/bash

# MMLU dataset
wget -O sdlm/data/instruction_evals/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p sdlm/data/instruction_evals/mmlu_data
tar -xvf sdlm/data/instruction_evals/mmlu_data.tar -C sdlm/data/instruction_evals/mmlu_data
rm -r sdlm/data/instruction_evals/mmlu_data.tar