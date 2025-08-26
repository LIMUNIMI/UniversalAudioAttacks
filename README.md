# Adversarial robustness evaluation of representation learning models and universal audio representations

Source code for the paper "Adversarial Robustness Evaluation of Representation Learning for Audio Classification".

## Setup

Use conda and the environment files provided as specified:

0. install base dependencies: `wget`, `tar`, `uv`
1. `uv sync`

## Run

- `uv run Dataset_import.py` - download the datasets
- `uv run Resample.py` - resamples the data
- `uv run Model_import.py` - compute and evaluate the embeddings
- `uv run Main_Loop.py` - perform the attacks and evaluate them
- `uv run SVM.py` - perform the SVM-based evaluation of the adversarial examples
- `uv run MLP.py` - perform the MLP-based evaluation of the adversarial examples

## Results

The results are presentend in the notebooks.  
For a direct access the two zip files contain the final results for the Attack and SVM phases.
