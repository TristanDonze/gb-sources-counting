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
    learning_strategy,
    MAX_K,
    BATCH_SIZE,
    LR,
    LR_MIN,
    WEIGHT_DECAY,
    NB_EPOCHS,
)

logger = logging.getLogger(__name__)


def train(run_manager, load_checkpoint_path=None):
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=5,
        T_mult=2,
        eta_min=LR_MIN,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=NB_EPOCHS,
    #     eta_min=LR_MIN,
    # )

    train_dataset = GalacticBinariesDataset(
        train_dataset_path, 
        max_K=MAX_K, 
        max_samples=1_000_000,
        noise=True,
        deterministic=False,
        seed=42,
    )

    val_dataset = GalacticBinariesDataset(
        val_dataset_path,
        max_K=MAX_K,
        max_samples=100_000,
        noise=False,
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

    checkpoint_dir = run_manager.checkpoint_dir

    for epoch in range(start_epoch, NB_EPOCHS):
        train_loss = train_one_epoch(
            model,
            MAX_K,
            train_loader,
            criterion,
            optimizer,
            scheduler,
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
                f"{checkpoint_dir}/best_checkpoint.pth",
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
                f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth",
            )
    logger.info(
        f"Training completed. Best Val F1: {best_val_f1:.4f} "
        f"at epoch {best_val_f1_epoch}"
    )
    run_manager.make_plot(
        name="Training Loss",
        values=train_losses,
        xlabel="Epoch",
        ylabel="Loss",
    )
    run_manager.make_plot(
        name="Validation Loss",
        values=val_losses,
        xlabel="Epoch",
        ylabel="Loss",
    )
    run_manager.make_plot(
        name="Validation Accuracy",
        values=val_accs,
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    run_manager.make_plot(
        name="Validation Recall",
        values=val_recall,
        xlabel="Epoch",
        ylabel="Recall",
    )
    run_manager.make_plot(
        name="Validation F1 Score",
        values=val_f1s,
        xlabel="Epoch",
        ylabel="F1 Score",
    )
    run_manager.make_plot(
        name="Validation MAE",
        values=val_maes,
        xlabel="Epoch",
        ylabel="MAE",
    )
    aim_run.close()
