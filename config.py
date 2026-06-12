import os
from pathlib import Path

# Dataset paths
# SNR : 10 - 100
small_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_200K_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
medium_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_uniform_SNR_10_100/dataset.hdf5")
large_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_3M_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
huge_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_10M_uniform_SNR_10_100/dataset.hdf5")

# SNR : 5 - 150
# small_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_200K_5.0e-24_2.0e-22_filtering_True_5_150_difficulty_1/dataset.hdf5")
# medium_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_5.0e-24_2.0e-22_filtering_True_5_150_difficulty_1/dataset.hdf5")
# large_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_3M_5.0e-24_2.0e-22_filtering_True_5_150_difficulty_1/dataset.hdf5")

TRAIN_SIZE = 0.5
MAX_SAMPLES_TRAIN = None
MAX_SAMPLES_VAL = 500_000

SPLIT_STRATEGY = "snr" # "random" or "snr"
SEED_TRAIN = 42
SEED_VAL = 0
SPLIT_SEED = 2027

# Training Hyperparameters 

learning_strategy = "mse" # "mse", "cross_entropy", "mse+ce" "ordinal"
WEIGHT_BY_K = True
MSE_K_WEIGHT_ALPHA = 1.0

LAMBDA_MSE = 1.0
LAMBDA_CE = 0.05

MAX_K = 10
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-3
NB_EPOCHS = 500

# Scheduler : 

LR = 2e-4
LR_MIN = 1e-6
FACTOR = 0.5
PATIENCE = 5
