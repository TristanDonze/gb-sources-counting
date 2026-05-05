import logging
import torch
from torch.utils.data import DataLoader

from src.model import CardinalityEstimator
from src.dataset import GalacticBinariesDataset

from src.training import train_one_epoch
from src.validation import evaluate
from src.utils import save_checkpoint, load_checkpoint
from src.aim_instance import aim_run, track_metric

from config import (
    train_dataset_path,
    val_dataset_path,
    BATCH_SIZE,
    LR,
    LR_MIN,
    WEIGHT_DECAY,
    NB_EPOCHS,
)

logger = logging.getLogger(__name__)


def train(load_checkpoint_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CardinalityEstimator().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=NB_EPOCHS,
        eta_min=LR_MIN,
    )

    train_dataset = GalacticBinariesDataset(train_dataset_path, max_K=10, max_samples=10_000)
    val_dataset = GalacticBinariesDataset(
        val_dataset_path,
        max_K=10,
        max_samples=1_000,
        deterministic=True,
        seed=0,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    aim_run["hparams"] = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "learning_rate_min": LR_MIN,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NB_EPOCHS,
    }
    aim_run["dataset"] = {
        "train_dataset_path": str(train_dataset_path),
        "val_dataset_path": str(val_dataset_path),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    train_losses = []
    val_losses = []
    val_accs = []
    val_recall = []
    val_f1s = []
    val_maes = []
    best_val_f1 = 0.0
    best_val_f1_epoch = 0

    if load_checkpoint_path is not None:
        (
            train_losses,
            val_losses,
            val_accs,
            val_recall,
            val_f1s,
            val_maes,
            best_val_f1,
            best_val_f1_epoch,
            last_completed_epoch,
        ) = load_checkpoint(model, optimizer, scheduler, load_checkpoint_path)
        start_epoch = last_completed_epoch + 1
        logger.info(f"Loaded checkpoint from {load_checkpoint_path}, starting from epoch {start_epoch+1}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, NB_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
        )
        val_loss, val_acc, val_recall_score, val_f1, mae = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_recall.append(val_recall_score)
        val_f1s.append(val_f1)
        val_maes.append(mae)

        aim_epoch = epoch + 1
        track_metric(
            "loss",
            train_loss,
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
            f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
            f"- Val Recall: {val_recall_score:.4f} - Val F1: {val_f1:.4f} - Val MAE: {mae:.4f}"
        )

        # Save checkpoint if current epoch has the best validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch + 1
            track_metric(
                "best_f1",
                best_val_f1,
                step=aim_epoch,
                epoch=aim_epoch,
                split="val",
                granularity="epoch",
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                train_losses,
                val_losses,
                val_accs,
                val_recall,
                val_f1s,
                val_maes,
                best_val_f1,
                best_val_f1_epoch,
                epoch,
                f"models/best_checkpoint_epoch_{best_val_f1_epoch}.pth",
            )
        else:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                train_losses,
                val_losses,
                val_accs,
                val_recall,
                val_f1s,
                val_maes,
                best_val_f1,
                best_val_f1_epoch,
                epoch,
                f"models/checkpoint_epoch_{epoch}.pth",
            )
    logger.info(
        f"Training completed. Best Val F1: {best_val_f1:.4f} "
        f"at epoch {best_val_f1_epoch}"
    )
    aim_run.close()
