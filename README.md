# Amplified Patch-Level Differential Privacy for Free via Random Cropping

This is the official implementation of our paper.

## Requirements
To install the requirements, execute:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Installation
You can install this package via `uv pip install -e .`

## Usage
In order to reproduce all experiments, you will need to execute the scripts in `seml/scripts` using the config files provided in `seml/configs` using the [SLURM Experiment Management Library](https://github.com/TUM-DAML/seml).  

After computing all results, you can use the scripts in `plotting` to recreate the figures from the paper.  

## Cite
Please cite our paper if you use this code in your own work.