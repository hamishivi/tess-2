# TESS v2

## Installation

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
# optional for devs
pip install pre-commit
pre-commit install
```

## Training

```sh
CUDA_VISIBLE_DEVICES=XYZ python sdlm/run_mlm.py configs/models/CONFIG.json
```
