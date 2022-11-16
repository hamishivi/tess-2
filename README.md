# Simplex Diffusion Language Model (SDLM).

# How to setup the environment
```
conda env create -f environment.yaml --prefix  /net/nfs.cirrascale/s2-research/rabeehk/conda/envs/sdlm
python setup develop
```
to update environment after installation:
```
conda env update --file environment.yaml --prune
```

# Process the data.
```
bash scripts/run_process_data.sh  configs/openwebtext.json
```
