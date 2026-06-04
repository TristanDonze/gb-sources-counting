import logging
import torch
from torch.utils.data import DataLoader

from src.model import CardinalityEstimator
from src.dataset import create_train_val_datasets

from src.training import train_one_epoch
from src.validation import evaluate
from src.utils import save_checkpoint, load_checkpoint
from src.aim_instance import aim_run, track_metric

from config import (
    small_dataset_path,
    medium_dataset_path,
    large_dataset_path,
    huge_dataset_path,
    TRAIN_SIZE,
    MAX_SAMPLES_TRAIN,
    MAX_SAMPLES_VAL,
    SPLIT_STRATEGY,
    SPLIT_SEED,
    SEED_TRAIN,
    SEED_VAL,
    learning_strategy,
    MAX_K,
    BATCH_SIZE,
    LR,
    LR_MIN,
    WEIGHT_DECAY,
    NB_EPOCHS,
    EARLY_STOPPING_PATIENCE_AFTER_MIN_LR,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=LR_MIN,
    )

    dataset_path = huge_dataset_path

    train_dataset, val_dataset, val_energy_matched_dataset = create_train_val_datasets(
        dataset_path,
        train_size=TRAIN_SIZE,
        max_K=MAX_K,
        max_samples_train=MAX_SAMPLES_TRAIN,
        max_samples_val=MAX_SAMPLES_VAL,
        noise_train=True,
        noise_val=True,
        deterministic_train=False,
        deterministic_val=True,
        split_seed=SPLIT_SEED,
        split_strategy=SPLIT_STRATEGY,
        seed_train=SEED_TRAIN,
        seed_val=SEED_VAL,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    val_energy_matched_loader = DataLoader(
        val_energy_matched_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
    )

    aim_run["hparams"] = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "learning_rate_min": LR_MIN,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 10,
        "early_stopping_patience_after_min_lr": EARLY_STOPPING_PATIENCE_AFTER_MIN_LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NB_EPOCHS,
    }
    aim_run["dataset"] = {
        "dataset_path": str(dataset_path),
        "train_size": TRAIN_SIZE,
        "split_seed": SPLIT_SEED,
        "train_waveforms": train_dataset.total_waveforms,
        "val_waveforms": val_dataset.total_waveforms,
        "val_energy_matched_waveforms": val_energy_matched_dataset.total_waveforms,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_energy_matched_samples": len(val_energy_matched_dataset),
    }

    train_losses = []
    val_losses = []
    val_accs = []
    val_recall = []
    val_f1s = []
    val_maes = []
    val_energy_matched_losses = []
    val_energy_matched_accs = []
    val_energy_matched_recall = []
    val_energy_matched_f1s = []
    val_energy_matched_maes = []
    best_val_f1 = 0.0
    best_val_f1_epoch = 0
    min_lr_reached_epoch = None

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
            val_energy_matched_losses,
            val_energy_matched_accs,
            val_energy_matched_recall,
            val_energy_matched_f1s,
            val_energy_matched_maes,
            min_lr_reached_epoch,
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
        (
            val_energy_matched_loss,
            val_energy_matched_acc,
            val_energy_matched_recall_score,
            val_energy_matched_f1,
            val_energy_matched_mae,
        ) = evaluate(
            model,
            MAX_K,
            val_energy_matched_loader,
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
        val_energy_matched_losses.append(val_energy_matched_loss)
        val_energy_matched_accs.append(val_energy_matched_acc)
        val_energy_matched_recall.append(val_energy_matched_recall_score)
        val_energy_matched_f1s.append(val_energy_matched_f1)
        val_energy_matched_maes.append(val_energy_matched_mae)

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
            "loss",
            val_energy_matched_loss,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val_energy_matched",
            granularity="epoch",
        )
        track_metric(
            "accuracy",
            val_energy_matched_acc,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val_energy_matched",
            granularity="epoch",
        )
        track_metric(
            "recall",
            val_energy_matched_recall_score,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val_energy_matched",
            granularity="epoch",
        )
        track_metric(
            "f1",
            val_energy_matched_f1,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val_energy_matched",
            granularity="epoch",
        )
        track_metric(
            "mae",
            val_energy_matched_mae,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val_energy_matched",
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
        logger.info(
            f"Epoch {epoch+1}/{NB_EPOCHS} - Val Energy-Matched Loss: {val_energy_matched_loss:.4f} "
            f"- Val Energy-Matched Acc: {val_energy_matched_acc:.4f} "
            f"- Val Energy-Matched Recall: {val_energy_matched_recall_score:.4f} "
            f"- Val Energy-Matched F1: {val_energy_matched_f1:.4f} "
            f"- Val Energy-Matched MAE: {val_energy_matched_mae:.4f}"
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
                val_energy_matched_losses=val_energy_matched_losses,
                val_energy_matched_accs=val_energy_matched_accs,
                val_energy_matched_recall=val_energy_matched_recall,
                val_energy_matched_f1s=val_energy_matched_f1s,
                val_energy_matched_maes=val_energy_matched_maes,
                min_lr_reached_epoch=min_lr_reached_epoch,
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
                val_energy_matched_losses=val_energy_matched_losses,
                val_energy_matched_accs=val_energy_matched_accs,
                val_energy_matched_recall=val_energy_matched_recall,
                val_energy_matched_f1s=val_energy_matched_f1s,
                val_energy_matched_maes=val_energy_matched_maes,
                min_lr_reached_epoch=min_lr_reached_epoch,
            )

        epochs_since_best_after_min_lr = (
            aim_epoch - max(best_val_f1_epoch, min_lr_reached_epoch or aim_epoch)
        )
        if (
            lr_at_min
            and epochs_since_best_after_min_lr >= EARLY_STOPPING_PATIENCE_AFTER_MIN_LR
        ):
            logger.info(
                "Early stopping triggered after %s epochs without Val F1 improvement "
                "at minimum LR %.2e. Best Val F1: %.4f at epoch %s.",
                epochs_since_best_after_min_lr,
                LR_MIN,
                best_val_f1,
                best_val_f1_epoch,
            )
            break
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
    run_manager.make_plot(
        name="Validation Energy-Matched Loss",
        values=val_energy_matched_losses,
        xlabel="Epoch",
        ylabel="Loss",
    )
    run_manager.make_plot(
        name="Validation Energy-Matched Accuracy",
        values=val_energy_matched_accs,
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    run_manager.make_plot(
        name="Validation Energy-Matched Recall",
        values=val_energy_matched_recall,
        xlabel="Epoch",
        ylabel="Recall",
    )
    run_manager.make_plot(
        name="Validation Energy-Matched F1 Score",
        values=val_energy_matched_f1s,
        xlabel="Epoch",
        ylabel="F1 Score",
    )
    run_manager.make_plot(
        name="Validation Energy-Matched MAE",
        values=val_energy_matched_maes,
        xlabel="Epoch",
        ylabel="MAE",
    )
    aim_run.close()
