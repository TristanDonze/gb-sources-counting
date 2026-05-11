import os
from pathlib import Path

# Dataset paths
train_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/train_dataset_1M_diff_1.hdf5")
val_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/val_dataset_100K_diff_1.hdf5")

# Models saving paths
model_save_dir = Path("/sps/l2it/tdonze/gb-source-counting/models")

# Training Hyperparameters 

learning_strategy = "mse" # "mse", "cross_entropy" or "ordinal"

MAX_K = 10

BATCH_SIZE = 256

LR = 3e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-2
NB_EPOCHS = 100