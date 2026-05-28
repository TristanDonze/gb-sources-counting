import os
from pathlib import Path

# Dataset paths

small_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_200K_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
medium_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
large_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_3M_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
# dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_variable_amp.hdf5")
train_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/train_dataset_1M_diff_1.hdf5")
val_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/val_dataset_100K_diff_1.hdf5")

TRAIN_SIZE = 0.8
SPLIT_SEED = 42

# Training Hyperparameters 

learning_strategy = "ordinal" # "mse", "cross_entropy" or "ordinal"

MAX_K = 10

BATCH_SIZE = 256

LR = 3e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-2
NB_EPOCHS = 75