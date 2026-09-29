# Galactic Binaries Source Counting

This repository is the second part of the work carried out during my internship at [L2IT](https://www.l2it.in2p3.fr/). It focuses on using deep learning to estimate the number of overlapping Galactic binaries in a signal.

The complete project is divided into four parts:

1. [Dataset generation](https://github.com/TristanDonze/gb-dataset-gen)
2. [Source counting](https://github.com/TristanDonze/gb-sources-counting)
3. [Source separation](https://github.com/TristanDonze/gb-sources-separation)
4. [Parameter estimation](https://github.com/TristanDonze/gb-parameters-estimation)

## Installation with uv

The project requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/TristanDonze/gb-sources-counting.git
cd gb-sources-counting
uv sync
```

`uv sync` creates the virtual environment and installs the dependencies declared in `pyproject.toml` and locked in `uv.lock`. On Linux, the project is configured to use the PyTorch packages for CUDA 12.8.

## Data and configuration

Training requires an HDF5 file containing the waveforms and their parameters. This dataset can be generated with the [gb-dataset-gen](https://github.com/TristanDonze/gb-dataset-gen) repository.

Before starting a training run, edit `config.py` to configure:

- the dataset path in `dataset_path`;
- the training and validation set sizes;
- the learning strategy and its hyperparameters;
- the maximum number of sources, `MAX_K`, and the number of epochs, `NB_EPOCHS`.

The dataset must contain a `waveforms` array and a `params` group. The SNR-based split strategy also requires a `params/snr` entry.

## Training

From the repository root, run:

```bash
uv run python main.py
```

Each training run creates a timestamped directory under `runs/`. It contains the checkpoints, including `best_checkpoint.pth`, the model definition used for the run, training plots, and reproducibility information. Metrics are also tracked with Aim.

To open the Aim interface during or after training, run:

```bash
uv run aim up
```

## Model evaluation

Evaluate the model by running the notebooks in the following order:

1. `notebooks/evaluate_model.ipynb` loads a checkpoint, evaluates the model, and produces a CSV file containing the results.
2. `notebooks/plot_results.ipynb` loads this CSV file and generates the analysis plots.

Before running all cells in `evaluate_model.ipynb`, update the following variables in its configuration cell:

- `run_name`, to select the `runs/run_...` directory to evaluate;
- `dataset_path`, along with the dataset paths imported from `config.py` if necessary;
- `learning_strategy`, so that it matches the strategy used during training.

Run all cells in `evaluate_model.ipynb`, followed by all cells in `plot_results.ipynb`. By default, both notebooks use the `evaluation_results_model_triple_head.csv` file located in the `notebooks` directory.
