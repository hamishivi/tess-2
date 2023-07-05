# TESS v2

## Installation

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# optional for devs
pip install pre-commit
pre-commit install
```

## Training

```sh
./shell_scripts/test.sh
```

## Beaker

Look at `shell_scripts/run_glue.sh` and `shell_scripts/run_pretrain.sh` for scripts that use [beaker-gantry](https://github.com/allenai/beaker-gantry) to run. These should be pretty easy to modify and run!

## Demo

Make sure the arguments in the script agree with how the model was trained...!

```sh
./shell_scripts/run_interactive_demo.sh
```