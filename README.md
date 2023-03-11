# Simplex Diffusion Language Model (SDLM)

## Installation

```sh
pip install -r requirements.txt
pip install -e .
# optional for devs
pip install pre-commit
pre-commit install
```

## Data

```
python sdlm/data/process_data.py configs/data/CONFIG.json
```

## Training

```sh
CUDA_VISIBLE_DEVICES=XYZ python sdlm/run_mlm.py configs/models/CONFIG.json
```
