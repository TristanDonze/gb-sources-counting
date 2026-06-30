import os
from pathlib import Path

# Dataset paths
# SNR : 10 - 100
small_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_200K_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
medium_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_uniform_SNR_10_100/dataset.hdf5")
large_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_3M_2.0e-23_1.0e-22_filtering_True_10_100_difficulty_1/dataset.hdf5")
huge_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_10M_uniform_SNR_10_100/dataset.hdf5")

# Medium Config
dataset_path = medium_dataset_path
TRAIN_SIZE = 0.8
MAX_SAMPLES_TRAIN = 1_000_000
MAX_SAMPLES_VAL = 200_000

# Huge Config
# dataset_path = huge_dataset_path
# TRAIN_SIZE = 0.5
# MAX_SAMPLES_TRAIN = None
# MAX_SAMPLES_VAL = 500_000

SPLIT_STRATEGY = "snr" # "random" or "snr"
SEED_TRAIN = 42
SEED_VAL = 0
SPLIT_SEED = 2027

# Training Hyperparameters 

learning_strategy = "mse+ce+or" # "mse", "cross_entropy", "mse+ce", "mse+ce+or", "ordinal"
HYBRID_STRATEGIES = {"mse+ce", "mse+ce+or"}
PRIMARY_PREDICTOR = "mse" # "mse", "ce", "ordinal", "ensemble"

WEIGHT_BY_K = True
MSE_K_WEIGHT_ALPHA = 1.0

LAMBDA_MSE = 100.0
LAMBDA_CE = 1.0
LAMBDA_ORDINAL = 0.03

LAMBDA_PREDICTION_MSE = 0.7
LAMBDA_PREDICTION_CE = 0.3 if learning_strategy == "mse+ce" else 0.29 if learning_strategy == "mse+ce+or" else 0.0
LAMBDA_PREDICTION_ORDINAL = 0.01

MAX_K = 10
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-3
NB_EPOCHS = 500

# Scheduler : 

LR = 2e-4
LR_MIN = 1e-6
FACTOR = 0.5
PATIENCE = 10
