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
    HYBRID_STRATEGIES,
    PRIMARY_PREDICTOR,
    MAX_K,
    BATCH_SIZE,
    WEIGHT_DECAY,
    NB_EPOCHS,
    LR,
    LR_MIN,
    FACTOR,
    PATIENCE,


    LAMBDA_MSE,
    LAMBDA_CE,
    LAMBDA_ORDINAL,

    LAMBDA_PREDICTION_MSE,
    LAMBDA_PREDICTION_CE,
    LAMBDA_PREDICTION_ORDINAL,

    WEIGHT_BY_K,
    MSE_K_WEIGHT_ALPHA,
)

logger = logging.getLogger(__name__)


def _unpack_eval_result(eval_result, learning_strategy):
    if learning_strategy not in HYBRID_STRATEGIES:
        loss, acc, recall_score, f1, mae = eval_result
        return {
            "loss": loss,
            "loss_mse": None,
            "loss_ce": None,
            "loss_ordinal": None,
            "primary": {
                "acc": acc,
                "recall": recall_score,
                "f1": f1,
                "mae": mae,
            },
            "predictors": None,
        }

    primary = eval_result["predictors"][PRIMARY_PREDICTOR]
    return {
        "loss": eval_result["loss"],
        "loss_mse": eval_result["loss_mse"],
        "loss_ce": eval_result["loss_ce"],
        "loss_ordinal": eval_result.get("loss_ordinal"),
        "primary": primary,
        "predictors": eval_result["predictors"],
    }


def _track_eval_metrics(metrics, *, step, epoch, split):
    track_metric(
        "loss",
        metrics["loss"],
        step=step,
        epoch=epoch,
        split=split,
        granularity="epoch",
    )

    if metrics["loss_mse"] is not None:
        track_metric(
            "loss_mse",
            metrics["loss_mse"],
            step=step,
            epoch=epoch,
            split=split,
            granularity="epoch",
        )
    if metrics["loss_ce"] is not None:
        track_metric(
            "loss_ce",
            metrics["loss_ce"],
            step=step,
            epoch=epoch,
            split=split,
            granularity="epoch",
        )
    if metrics["loss_ordinal"] is not None:
        track_metric(
            "loss_ordinal",
            metrics["loss_ordinal"],
            step=step,
            epoch=epoch,
            split=split,
            granularity="epoch",
        )

    if metrics["predictors"] is None:
        primary = metrics["primary"]
        for name, value in primary.items():
            track_metric(
                name,
                value,
                step=step,
                epoch=epoch,
                split=split,
                granularity="epoch",
            )
        return

    for predictor, predictor_metrics in metrics["predictors"].items():
        for name, value in predictor_metrics.items():
            track_metric(
                name,
                value,
                step=step,
                epoch=epoch,
                split=split,
                granularity="epoch",
                predictor=predictor,
            )


def _format_eval_metrics(label, metrics):
    if metrics["predictors"] is None:
        primary = metrics["primary"]
        return (
            f"{label}\n"
            f"  - Loss: {metrics['loss']:.4f}\n"
            f"  - F1: {primary['f1']:.4f}\n"
            f"  - MAE: {primary['mae']:.4f}\n"
        )
        

    parts = [
        f"{label} :\n",
        f"  - Loss: {metrics['loss']:.4f}\n",
        f"  - MSE Loss: {metrics['loss_mse']:.4f}\n",
        f"  - CE Loss: {metrics['loss_ce']:.4f}\n",
    ]
    if metrics["loss_ordinal"] is not None:
        parts.append(f"  - Ordinal Loss: {metrics['loss_ordinal']:.4f}\n")
    for predictor, predictor_metrics in metrics["predictors"].items():
        parts.append(
            f" Predictor: {predictor}\n"
            f"  - F1: {predictor_metrics['f1']:.4f}\n"
            f"  - MAE: {predictor_metrics['mae']:.4f}\n"
        )
    return "".join(parts)

def train(run_manager, load_checkpoint_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if learning_strategy == "mse":
        criterion = torch.nn.MSELoss()
    elif learning_strategy == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif learning_strategy == "ordinal":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif learning_strategy == "mse+ce":
        criterion = (
            torch.nn.MSELoss(), 
            torch.nn.CrossEntropyLoss()
        )
    elif learning_strategy == "mse+ce+or":
        criterion = (
            torch.nn.MSELoss(),
            torch.nn.CrossEntropyLoss(),
            torch.nn.BCEWithLogitsLoss(),
        )
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

    dataset_path = huge_dataset_path

    train_dataset, val_dataset, val_energy_matched_dataset = create_train_val_datasets(
        dataset_path,
        train_size=TRAIN_SIZE,
        max_K=MAX_K,
        max_samples_train=MAX_SAMPLES_TRAIN,
        max_samples_val=MAX_SAMPLES_VAL,
        noise_train=True,
        noise_val=True,
        random_global_scale_train=True,
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
        "scheduler_factor": FACTOR,
        "scheduler_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NB_EPOCHS,
        "learning_strategy": learning_strategy,
        "weight_by_K": WEIGHT_BY_K if learning_strategy == "mse" else None,
        "lambda_mse": LAMBDA_MSE if learning_strategy in HYBRID_STRATEGIES else None,
        "lambda_ce": LAMBDA_CE if learning_strategy in HYBRID_STRATEGIES else None,
        "lambda_ordinal": LAMBDA_ORDINAL if learning_strategy == "mse+ce+or" else None,
        "lambda_prediction_mse": LAMBDA_PREDICTION_MSE if learning_strategy in HYBRID_STRATEGIES else None,
        "lambda_prediction_ce": LAMBDA_PREDICTION_CE if learning_strategy in HYBRID_STRATEGIES else None,
        "lambda_prediction_ordinal": LAMBDA_PREDICTION_ORDINAL if learning_strategy == "mse+ce+or" else None,
        "primary_predictor": (
            PRIMARY_PREDICTOR
            if learning_strategy in HYBRID_STRATEGIES
            else learning_strategy
        ),
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
        val_result = evaluate(
            model,
            MAX_K,
            val_loader,
            criterion,
            learning_strategy,
            device,
        )
        val_metrics = _unpack_eval_result(val_result, learning_strategy)

        val_energy_matched_result = evaluate(
            model,
            MAX_K,
            val_energy_matched_loader,
            criterion,
            learning_strategy,
            device,
        )
        val_energy_matched_metrics = _unpack_eval_result(
            val_energy_matched_result,
            learning_strategy,
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["primary"]["acc"]
        val_recall_score = val_metrics["primary"]["recall"]
        val_f1 = val_metrics["primary"]["f1"]
        mae = val_metrics["primary"]["mae"]
        val_energy_matched_loss = val_energy_matched_metrics["loss"]
        val_energy_matched_acc = val_energy_matched_metrics["primary"]["acc"]
        val_energy_matched_recall_score = val_energy_matched_metrics["primary"]["recall"]
        val_energy_matched_f1 = val_energy_matched_metrics["primary"]["f1"]
        val_energy_matched_mae = val_energy_matched_metrics["primary"]["mae"]

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
        _track_eval_metrics(
            val_metrics,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val",
        )
        _track_eval_metrics(
            val_energy_matched_metrics,
            step=aim_epoch,
            epoch=aim_epoch,
            split="val_energy_matched",
        )
        track_metric(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            step=aim_epoch,
            epoch=aim_epoch,
            granularity="epoch",
        )

        logger.info(
            f"Epoch {epoch+1}/{NB_EPOCHS} - Train Loss: {train_loss:.4f}\n"
            f"{_format_eval_metrics('Val', val_metrics)}\n"
            f"{_format_eval_metrics('Val Energy-Matched', val_energy_matched_metrics)}"
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
                predictor=(
                    PRIMARY_PREDICTOR
                    if learning_strategy in HYBRID_STRATEGIES
                    else None
                ),
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
