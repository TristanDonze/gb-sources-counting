import logging

import torch
import wandb
from torch.utils.data import DataLoader

import config as default_config
from src.dataset import create_train_val_datasets
from src.model import CardinalityEstimator
from src.training import train_one_epoch
from src.utils import load_checkpoint, save_checkpoint
from src.validation import evaluate
from src.wandb_tracking import init_wandb_run, track_metric

logger = logging.getLogger(__name__)


def _is_hybrid_strategy(learning_strategy):
    return learning_strategy in {"mse+ce", "mse+ce+or"}


def _uses_ordinal_head(learning_strategy):
    return learning_strategy == "mse+ce+or"


def _build_run_config(config_overrides=None):
    overrides = dict(config_overrides or {})
    cfg = {
        "DATASET_PATH": default_config.DATASET_PATH,
        "TRAIN_SIZE": default_config.TRAIN_SIZE,
        "MAX_SAMPLES_TRAIN": default_config.MAX_SAMPLES_TRAIN,
        "MAX_SAMPLES_VAL": default_config.MAX_SAMPLES_VAL,
        "SPLIT_STRATEGY": default_config.SPLIT_STRATEGY,
        "SPLIT_SEED": default_config.SPLIT_SEED,
        "SEED_TRAIN": default_config.SEED_TRAIN,
        "SEED_VAL": default_config.SEED_VAL,
        "LEARNING_STRATEGY": default_config.LEARNING_STRATEGY,
        "PRIMARY_PREDICTOR": default_config.PRIMARY_PREDICTOR,
        "MAX_K": default_config.MAX_K,
        "BATCH_SIZE": default_config.BATCH_SIZE,
        "WEIGHT_DECAY": default_config.WEIGHT_DECAY,
        "NB_EPOCHS": default_config.NB_EPOCHS,
        "LR": default_config.LR,
        "LR_MIN": default_config.LR_MIN,
        "FACTOR": default_config.FACTOR,
        "PATIENCE": default_config.PATIENCE,
        "LAMBDA_MSE": default_config.LAMBDA_MSE,
        "LAMBDA_CE": default_config.LAMBDA_CE,
        "LAMBDA_ORDINAL": default_config.LAMBDA_ORDINAL,
        "LAMBDA_PREDICTION_MSE": default_config.LAMBDA_PREDICTION_MSE,
        "LAMBDA_PREDICTION_CE": default_config.LAMBDA_PREDICTION_CE,
        "LAMBDA_PREDICTION_ORDINAL": default_config.LAMBDA_PREDICTION_ORDINAL,
        "WEIGHT_BY_K": default_config.WEIGHT_BY_K,
        "MSE_K_WEIGHT_ALPHA": default_config.MSE_K_WEIGHT_ALPHA,
        "DIM_MODEL": default_config.DIM_MODEL,
        "CHANNEL_MULTIPLIER": default_config.CHANNEL_MULTIPLIER,
        "CONV_1_KERNEL_SIZE": default_config.CONV_1_KERNEL_SIZE,
        "CONV_2_KERNEL_SIZE": default_config.CONV_2_KERNEL_SIZE,
        "CONV_3_KERNEL_SIZE": default_config.CONV_3_KERNEL_SIZE,
        "CONV_4_KERNEL_SIZE": default_config.CONV_4_KERNEL_SIZE,
        "CONV_2_STRIDE": default_config.CONV_2_STRIDE,
        "TRANSFORMER_NHEAD": default_config.TRANSFORMER_NHEAD,
        "TRANSFORMER_FF_MULTIPLIER": default_config.TRANSFORMER_FF_MULTIPLIER,
        "TRANSFORMER_NUM_LAYERS": default_config.TRANSFORMER_NUM_LAYERS,
    }

    if wandb.run is not None:
        for key, value in dict(wandb.config).items():
            if key in cfg:
                cfg[key] = value

    for key, value in overrides.items():
        if key in cfg:
            cfg[key] = value
    return cfg


def _unpack_eval_result(eval_result, learning_strategy, primary_predictor):
    if not _is_hybrid_strategy(learning_strategy):
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

    primary = eval_result["predictors"][primary_predictor]
    return {
        "loss": eval_result["loss"],
        "loss_mse": eval_result["loss_mse"],
        "loss_ce": eval_result["loss_ce"],
        "loss_ordinal": eval_result.get("loss_ordinal"),
        "primary": primary,
        "predictors": eval_result["predictors"],
    }


def _track_eval_metrics(metrics, *, step, epoch, split):
    track_metric("loss", metrics["loss"], step=step, epoch=epoch, split=split, granularity="epoch")

    if metrics["loss_mse"] is not None:
        track_metric("loss_mse", metrics["loss_mse"], step=step, epoch=epoch, split=split, granularity="epoch")
    if metrics["loss_ce"] is not None:
        track_metric("loss_ce", metrics["loss_ce"], step=step, epoch=epoch, split=split, granularity="epoch")
    if metrics["loss_ordinal"] is not None:
        track_metric("loss_ordinal", metrics["loss_ordinal"], step=step, epoch=epoch, split=split, granularity="epoch")

    if metrics["predictors"] is None:
        primary = metrics["primary"]
        for name, value in primary.items():
            track_metric(name, value, step=step, epoch=epoch, split=split, granularity="epoch")
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


def train(load_checkpoint_path=None, config_overrides=None, run_name=None, project=None, entity=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")

    cfg = _build_run_config(config_overrides)
    learning_strategy = cfg["LEARNING_STRATEGY"]
    primary_predictor = cfg["PRIMARY_PREDICTOR"]

    cfg.update({
        "LAMBDA_MSE": cfg["LAMBDA_MSE"] if _is_hybrid_strategy(learning_strategy) else None,
        "LAMBDA_CE": cfg["LAMBDA_CE"] if _is_hybrid_strategy(learning_strategy) else None,
        "LAMBDA_ORDINAL": cfg["LAMBDA_ORDINAL"] if _uses_ordinal_head(learning_strategy) else None,
        "LAMBDA_PREDICTION_MSE": cfg["LAMBDA_PREDICTION_MSE"] if _is_hybrid_strategy(learning_strategy) else None,
        "LAMBDA_PREDICTION_CE": cfg["LAMBDA_PREDICTION_CE"] if _is_hybrid_strategy(learning_strategy) else None,
        "LAMBDA_PREDICTION_ORDINAL": cfg["LAMBDA_PREDICTION_ORDINAL"] if _uses_ordinal_head(learning_strategy) else None,
        "PRIMARY_PREDICTOR": primary_predictor if _is_hybrid_strategy(learning_strategy) else learning_strategy,
    })
    
    run = init_wandb_run(
        run_name=run_name,
        project=project,
        entity=entity,
        config=cfg,
    )

    checkpoint_dir = run.dir

    if learning_strategy == "mse":
        criterion = torch.nn.MSELoss()
    elif learning_strategy == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif learning_strategy == "ordinal":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif learning_strategy == "mse+ce":
        criterion = (torch.nn.MSELoss(), torch.nn.CrossEntropyLoss())
    elif learning_strategy == "mse+ce+or":
        criterion = (
            torch.nn.MSELoss(),
            torch.nn.CrossEntropyLoss(),
            torch.nn.BCEWithLogitsLoss(),
        )
    else:
        raise ValueError(f"Unknown learning strategy: {learning_strategy}")

    model = CardinalityEstimator(
        learning_strategy=learning_strategy,
        max_K=cfg["MAX_K"],
        dim_model=cfg["DIM_MODEL"],
        channel_multiplier=cfg["CHANNEL_MULTIPLIER"],
        conv_1_kernel_size=cfg["CONV_1_KERNEL_SIZE"],
        conv_2_kernel_size=cfg["CONV_2_KERNEL_SIZE"],
        conv_3_kernel_size=cfg["CONV_3_KERNEL_SIZE"],
        conv_4_kernel_size=cfg["CONV_4_KERNEL_SIZE"],
        conv_2_stride=cfg["CONV_2_STRIDE"],
        transformer_nhead=cfg["TRANSFORMER_NHEAD"],
        transformer_ff_multiplier=cfg["TRANSFORMER_FF_MULTIPLIER"],
        transformer_num_layers=cfg["TRANSFORMER_NUM_LAYERS"],
    ).to(device)
    logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.debug("Model architecture:")
    for name, module in model.named_modules():
        logger.debug(f"  {name}: {module}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=cfg["FACTOR"],
        patience=cfg["PATIENCE"],
        min_lr=cfg["LR_MIN"],
    )

    train_dataset, val_dataset, val_energy_matched_dataset = create_train_val_datasets(
        cfg["DATASET_PATH"],
        train_size=cfg["TRAIN_SIZE"],
        max_K=cfg["MAX_K"],
        max_samples_train=cfg["MAX_SAMPLES_TRAIN"],
        max_samples_val=cfg["MAX_SAMPLES_VAL"],
        noise_train=True,
        noise_val=True,
        random_global_scale_train=True,
        deterministic_train=False,
        deterministic_val=True,
        split_seed=cfg["SPLIT_SEED"],
        split_strategy=cfg["SPLIT_STRATEGY"],
        seed_train=cfg["SEED_TRAIN"],
        seed_val=cfg["SEED_VAL"],
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], num_workers=0, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg["BATCH_SIZE"], num_workers=0, shuffle=False)
    val_energy_matched_loader = DataLoader(
        val_energy_matched_dataset,
        batch_size=cfg["BATCH_SIZE"],
        num_workers=0,
        shuffle=False,
    )

    wandb.config.update(
        {
            "NB_TRAIN_WAVEFORMS": train_dataset.total_waveforms,
            "NB_VAL_WAVEFORMS": val_dataset.total_waveforms,
            "NB_VAL_ENERGY_MATCHED_WAVEFORMS": val_energy_matched_dataset.total_waveforms,
            "NB_TRAIN_SAMPLES": len(train_dataset),
            "NB_VAL_SAMPLES": len(val_dataset),
            "NB_VAL_ENERGY_MATCHED_SAMPLES": len(val_energy_matched_dataset),
        }
    )

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
        logger.info(f"Loaded checkpoint from {load_checkpoint_path}, starting from epoch {start_epoch + 1}")
    else:
        start_epoch = 0

    try:
        for epoch in range(start_epoch, cfg["NB_EPOCHS"]):
            epoch_num = epoch + 1
            logger.info('-' * 70)
            logger.info(f'Training Epoch {epoch_num}/{cfg["NB_EPOCHS"]} ...')
            train_loss = train_one_epoch(
                model,
                cfg["MAX_K"],
                train_loader,
                criterion,
                optimizer,
                learning_strategy,
                device,
                lambda_mse=cfg["LAMBDA_MSE"],
                lambda_ce=cfg["LAMBDA_CE"],
                lambda_ordinal=cfg["LAMBDA_ORDINAL"],
                weight_by_k=cfg["WEIGHT_BY_K"],
                mse_k_weight_alpha=cfg["MSE_K_WEIGHT_ALPHA"],
            )
            logger.info(
                f'Train Summary | Epoch {epoch_num} | Loss={train_loss:.4f}')

            val_result = evaluate(
                model,
                cfg["MAX_K"],
                val_loader,
                criterion,
                learning_strategy,
                device,
                lambda_mse=cfg["LAMBDA_MSE"],
                lambda_ce=cfg["LAMBDA_CE"],
                lambda_ordinal=cfg["LAMBDA_ORDINAL"],
                lambda_prediction_mse=cfg["LAMBDA_PREDICTION_MSE"],
                lambda_prediction_ce=cfg["LAMBDA_PREDICTION_CE"],
                lambda_prediction_ordinal=cfg["LAMBDA_PREDICTION_ORDINAL"],
            )
            val_metrics = _unpack_eval_result(val_result, learning_strategy, primary_predictor)

            val_energy_matched_result = evaluate(
                model,
                cfg["MAX_K"],
                val_energy_matched_loader,
                criterion,
                learning_strategy,
                device,
                lambda_mse=cfg["LAMBDA_MSE"],
                lambda_ce=cfg["LAMBDA_CE"],
                lambda_ordinal=cfg["LAMBDA_ORDINAL"],
                lambda_prediction_mse=cfg["LAMBDA_PREDICTION_MSE"],
                lambda_prediction_ce=cfg["LAMBDA_PREDICTION_CE"],
                lambda_prediction_ordinal=cfg["LAMBDA_PREDICTION_ORDINAL"],
            )
            val_energy_matched_metrics = _unpack_eval_result(
                val_energy_matched_result,
                learning_strategy,
                primary_predictor,
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

            scheduler.step(mae)

            lr_at_min = all(param_group["lr"] <= cfg["LR_MIN"] + 1e-12 for param_group in optimizer.param_groups)
            if lr_at_min and min_lr_reached_epoch is None:
                min_lr_reached_epoch = epoch_num
                logger.info(f"Minimum LR {cfg['LR_MIN']:.2e} reached at epoch {epoch_num}.")

            track_metric("loss", train_loss, step=epoch_num, epoch=epoch_num, split="train", granularity="epoch")
            _track_eval_metrics(val_metrics, step=epoch_num, epoch=epoch_num, split="val")
            _track_eval_metrics(val_energy_matched_metrics, step=epoch_num, epoch=epoch_num, split="val_energy_matched")
            track_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch_num, epoch=epoch_num, granularity="epoch")

            logger.info(
                f'Valid Summary | Epoch {epoch_num} | Loss={val_loss:.4f} | F1={val_f1:.4f} | MAE={mae:.4f}'
                )

            logger.debug(
                f"{_format_eval_metrics('Val', val_metrics)}\n"
                f"{_format_eval_metrics('Val Energy-Matched', val_energy_matched_metrics)}"
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_f1_epoch = epoch_num
                track_metric(
                    "best_f1",
                    best_val_f1,
                    step=epoch_num,
                    epoch=epoch_num,
                    split="val",
                    granularity="epoch",
                    predictor=primary_predictor if _is_hybrid_strategy(learning_strategy) else None,
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
    finally:
        if wandb.run is not None:
            wandb.run.summary["best_val_f1"] = best_val_f1
            wandb.run.summary["best_val_f1_epoch"] = best_val_f1_epoch
            wandb.run.finish()

    logger.info(f"Training completed. Best Val F1: {best_val_f1:.4f} at epoch {best_val_f1_epoch}")
