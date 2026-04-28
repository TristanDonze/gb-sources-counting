import os
from pathlib import Path

# Dataset paths
train_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/train_dataset.hdf5")
val_dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/val_dataset.hdf5")

# Models saving paths
model_save_dir = Path("/sps/l2it/tdonze/gb-source-counting/models")
best_model_save_path = model_save_dir / "best_model.pth"