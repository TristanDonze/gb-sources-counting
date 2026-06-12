import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

dataset_path = Path("/sps/l2it/tdonze/gb-dataset-gen/data/synthetic_dataset/dataset_500K_uniform_SNR_10_100/dataset.hdf5")

TRAIN_SIZE = 0.8
MAX_SAMPLES_TRAIN = 50_000
MAX_SAMPLES_VAL = None

SPLIT_STRATEGY = "snr"
SEED_TRAIN = 42
SEED_VAL = 0
SPLIT_SEED = 2027

# Training Hyperparameters 

learning_strategy = "ordinal" # "mse", "cross_entropy" or "ordinal"
MAX_K = 10
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-3
NB_EPOCHS = 100

# Scheduler : 

LR = 1e-3
LR_MIN = 1e-7
FACTOR = 0.5
PATIENCE = 10

import logging
import torch
from torch.utils.data import DataLoader

from src.model import CardinalityEstimator
from src.dataset import create_train_val_datasets

from src.training import train_one_epoch
from src.validation import evaluate
from src.aim_instance import aim_run, track_metric

logger = logging.getLogger(__name__)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if learning_strategy == "mse":
        criterion = torch.nn.MSELoss()
    elif learning_strategy == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif learning_strategy == "ordinal":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown learning strategy: {learning_strategy}")
    logger.info(f"Learning strategy: {learning_strategy}")

    model = CardinalityEstimator(learning_strategy=learning_strategy, max_K=MAX_K).to(device)
    logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("Model architecture:")
    for name, module in model.named_modules():
        logger.info(f"  {name}: {module}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=FACTOR,
        patience=PATIENCE,
        min_lr=LR_MIN,
    )

    train_dataset, val_dataset, _ = create_train_val_datasets(
        dataset_path,
        train_size=TRAIN_SIZE,
        max_K=MAX_K,
        max_samples_train=MAX_SAMPLES_TRAIN,
        max_samples_val=MAX_SAMPLES_VAL,
        noise_train=True,
        noise_val=True,
        deterministic_train=True,
        deterministic_val=True,
        split_seed=SPLIT_SEED,
        split_strategy=SPLIT_STRATEGY,
        seed_train=SEED_TRAIN,
        seed_val=SEED_VAL,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    # title of the run in Aim
    aim_run.name = f"overfitting_test_{dataset_path.stem}"

    aim_run["hparams"] = {
        "test_overfitting": True,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "learning_rate_min": LR_MIN,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": FACTOR,
        "scheduler_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NB_EPOCHS,
    }
    aim_run["dataset"] = {
        "dataset_path": str(dataset_path),
        "train_size": TRAIN_SIZE,
        "split_seed": SPLIT_SEED,
        "train_waveforms": train_dataset.total_waveforms,
        "val_waveforms": val_dataset.total_waveforms,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    train_losses = []
    train_losses_evaluation = []
    train_accs = []
    train_recall = []
    train_f1s = []
    train_maes = []
    val_losses = []
    val_accs = []
    val_recall = []
    val_f1s = []
    val_maes = []

    min_lr_reached_epoch = None
    
    for epoch in range(NB_EPOCHS):
        train_loss = train_one_epoch(
            model,
            MAX_K,
            train_loader,
            criterion,
            optimizer,
            learning_strategy,
            False,
            device,
        )
        train_loss_evaluation, train_acc, train_recall_score, train_f1, train_mae = evaluate(
            model,
            MAX_K,
            train_loader,
            criterion,
            learning_strategy,
            device,
        )
        val_loss, val_acc, val_recall_score, val_f1, mae = evaluate(
            model,
            MAX_K,
            val_loader,
            criterion,
            learning_strategy,
            device,
        )
        train_losses.append(train_loss)
        train_losses_evaluation.append(train_loss_evaluation)
        train_accs.append(train_acc)
        train_recall.append(train_recall_score)
        train_f1s.append(train_f1)
        train_maes.append(train_mae)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_recall.append(val_recall_score)
        val_f1s.append(val_f1)
        val_maes.append(mae)

        scheduler.step(val_loss)

        aim_epoch = epoch + 1
        lr_at_min = all(
            param_group["lr"] <= LR_MIN + 1e-12
            for param_group in optimizer.param_groups
        )
        if lr_at_min and min_lr_reached_epoch is None:
            min_lr_reached_epoch = aim_epoch
            logger.info(f"Minimum LR {LR_MIN:.2e} reached at epoch {aim_epoch}.")

        track_metric(
            "loss",
            train_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "train_loss_evaluation",
            train_loss_evaluation,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "accuracy",
            train_acc,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "recall",
            train_recall_score,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "f1",
            train_f1,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )
        track_metric(
            "mae",
            train_mae,
            step=aim_epoch,
            epoch=aim_epoch,
            split="train",
            granularity="epoch",
        )

        track_metric(
            "loss",
            val_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "accuracy",
            val_acc,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "recall",
            val_recall_score,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "f1",
            val_f1,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )
        track_metric(
            "mae",
            mae,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
            granularity="epoch",
        )

        track_metric(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            step=aim_epoch,
            epoch=aim_epoch,
            granularity="epoch",
        )

        logger.info(
            f"Epoch {epoch+1}/{NB_EPOCHS} - Train Loss: {train_loss:.4f} "
            f"- Train Loss (Eval): {train_loss_evaluation:.4f} - Train Acc: {train_acc:.4f} - Train Recall: {train_recall[-1]:.4f} - Train F1: {train_f1:.4f} - Train MAE: {train_mae:.4f} "
            f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
            f"- Val Recall: {val_recall_score:.4f} - Val F1: {val_f1:.4f} - Val MAE: {mae:.4f}"
        )
    logger.info("Overfitting test completed.")

if __name__ == "__main__":
    train()
    aim_run.close()
