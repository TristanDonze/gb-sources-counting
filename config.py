import os
from pathlib import Path

# Dataset paths
train_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/ train_dataset_diff_1.hdf5")
val_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/ val_dataset_diff_1.hdf5")

# Models saving paths
model_save_dir = Path("/sps/l2it/tdonze/gb-source-counting/models")

# Training Hyperparameters 

BATCH_SIZE = 128

LR = 3e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-2
NB_EPOCHS = 50