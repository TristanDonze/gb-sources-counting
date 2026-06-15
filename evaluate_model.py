import csv
import gc
import gzip
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    MAX_K,
    SPLIT_SEED,
    SPLIT_STRATEGY,
    TRAIN_SIZE,
    learning_strategy,
    huge_dataset_path,
)
from src.dataset import split_dataset_indices, snr_based_split_dataset_indices
from src.val_dataset import HomogeneousSNRValidationDataset, SNRGapValidationDataset


DATASET_PATH = huge_dataset_path
RUN_PATH = Path("runs/run_20260611_145257/checkpoints")
MODEL_STRUCTURE_PATH = RUN_PATH / "model_structure.py"
CHECKPOINT_PATH = RUN_PATH / "best_checkpoint.pth"

SNR_GAP_TARGETS = [10, 20, 30, 40, 50, 60, 70, 80]
HOMOGENEOUS_TARGETS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 98]
SEEDS = [84029459]

MAX_SAMPLES = 200_000
BATCH_SIZE = 512
HOMOGENEOUS_POOL_SIZE = 500_000
MAX_GAP_RETRIES = 1_000
NOISE = True
SAVE_SAMPLE_DETAILS = True

OUTPUT_DIR = Path("evaluation_results") / f"snr_validation_{datetime.now():%Y%m%d_%H%M%S}"
VAL_INDICES = None


def get_validation_indices():
    with h5py.File(DATASET_PATH, "r") as f:
        total_waveforms = f["waveforms"].shape[0]
        if SPLIT_STRATEGY == "random":
            _, val_indices = split_dataset_indices(
                total_waveforms=total_waveforms,
                train_size=TRAIN_SIZE,
                split_seed=SPLIT_SEED,
            )
        elif SPLIT_STRATEGY == "snr":
            _, val_indices = snr_based_split_dataset_indices(
                snr_values=f["params"]["snr"][:],
                train_size=TRAIN_SIZE,
                split_seed=SPLIT_SEED,
            )
        else:
            raise ValueError("SPLIT_STRATEGY must be either 'random' or 'snr'")

    return val_indices


def load_model(device):
    spec = importlib.util.spec_from_file_location("dynamic_model", MODEL_STRUCTURE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_model"] = module
    spec.loader.exec_module(module)

    model_cls = getattr(module, "CardinalityEstimator")
    model = model_cls(
        learning_strategy=learning_strategy,
        max_K=MAX_K,
        input_channels=4,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_dataset(kind, target, seed):
    common = dict(
        dataset_path=DATASET_PATH,
        max_K=MAX_K,
        max_samples=MAX_SAMPLES,
        noise=NOISE,
        seed=seed,
        return_params=True,
        indices=VAL_INDICES,
    )

    if kind == "snr_gap":
        return SNRGapValidationDataset(
            target_snr_gap=target,
            max_retries=MAX_GAP_RETRIES,
            **common,
        )

    if kind == "homogeneous":
        return HomogeneousSNRValidationDataset(
            target_snr=target,
            pool_size=HOMOGENEOUS_POOL_SIZE,
            **common,
        )

    raise ValueError(f"Unknown dataset kind: {kind}")


def get_predictions_and_loss(logits, labels, device):
    if learning_strategy == "ordinal":
        thresholds = torch.arange(1, MAX_K, device=device)
        y = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
        probs = torch.sigmoid(logits)
        pred = 1 + (probs > 0.5).sum(dim=1)
        pred = pred.clamp(1, MAX_K)
        pred_score = 1.0 + probs.sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(dim=1)
        return pred, pred_score, loss

    if learning_strategy == "cross_entropy":
        y = labels - 1
        probs = F.softmax(logits, dim=1)
        class_values = torch.arange(1, MAX_K + 1, device=device).float()
        pred = logits.argmax(dim=1) + 1
        pred_score = (probs * class_values.unsqueeze(0)).sum(dim=1)
        loss = F.cross_entropy(logits, y, reduction="none")
        return pred, pred_score, loss

    if learning_strategy == "mse":
        y = labels.float().unsqueeze(1) / MAX_K
        out = torch.sigmoid(logits)
        pred_score = out.squeeze(1) * MAX_K
        pred = torch.round(pred_score).long().clamp(1, MAX_K)
        loss = F.mse_loss(out, y, reduction="none").squeeze(1)
        return pred, pred_score, loss

    raise ValueError(f"Unknown learning strategy: {learning_strategy}")


def compute_snr_stats(params, x):
    snr = torch.as_tensor(params["snr"], dtype=torch.float32, device=x.device)
    mask = torch.as_tensor(params["source_mask"], dtype=torch.bool, device=x.device)
    count = mask.sum(dim=1).clamp(min=1)

    snr_zero = torch.where(mask, snr, torch.zeros_like(snr))
    snr_inf = torch.where(mask, snr, torch.full_like(snr, float("inf")))
    snr_mean = snr_zero.sum(dim=1) / count

    return {
        "snr_mix": torch.sqrt((snr_zero ** 2).sum(dim=1)),
        "snr_min": snr_inf.min(dim=1).values,
        "snr_max": snr_zero.max(dim=1).values,
        "snr_mean": snr_mean,
        "snr_gap": snr_zero.max(dim=1).values - snr_inf.min(dim=1).values,
        "energy_noisy": (x ** 2).sum(dim=(1, 2)),
    }


def write_sample_rows(writer, kind, target, seed, offset, labels, pred, pred_score, loss, stats):
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()
    pred_score = pred_score.cpu().numpy()
    loss = loss.cpu().numpy()
    stats = {key: value.cpu().numpy() for key, value in stats.items()}

    for i in range(len(labels)):
        error = int(pred[i] - labels[i])
        writer.writerow(
            {
                "dataset": kind,
                "target": target,
                "seed": seed,
                "idx": offset + i,
                "true_K": int(labels[i]),
                "pred_K": int(pred[i]),
                "pred_score": float(pred_score[i]),
                "loss": float(loss[i]),
                "correct": bool(pred[i] == labels[i]),
                "error_K": error,
                "abs_error_K": abs(error),
                **{key: float(value[i]) for key, value in stats.items()},
            }
        )


def evaluate_one(model, dataset, kind, target, seed, device, writer):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    y_true, y_pred, losses = [], [], []
    offset = 0

    with torch.no_grad():
        for waveforms, labels, params in tqdm(
            loader,
            desc=f"{kind} target={target} seed={seed}",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ):
            x = torch.as_tensor(waveforms, dtype=torch.float32, device=device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            logits = model(x)
            pred, pred_score, loss = get_predictions_and_loss(logits, labels, device)
            snr_stats = compute_snr_stats(params, x)

            if writer is not None:
                write_sample_rows(
                    writer,
                    kind,
                    target,
                    seed,
                    offset,
                    labels,
                    pred,
                    pred_score,
                    loss,
                    snr_stats,
                )

            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            losses.append(loss.cpu().numpy())
            offset += labels.shape[0]

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    losses = np.concatenate(losses)
    labels = np.arange(1, MAX_K + 1)

    summary = {
        "dataset": kind,
        "target": target,
        "seed": seed,
        "n": len(y_true),
        "loss": float(losses.mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
        "mean_error": float(np.mean(y_pred - y_true)),
        **{f"dataset_{key}": value for key, value in dataset.stats.items()},
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return summary, cm


def save_summary(rows):
    path = OUTPUT_DIR / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global VAL_INDICES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    VAL_INDICES = get_validation_indices()

    metadata = {
        "dataset_path": str(DATASET_PATH),
        "model_structure_path": str(MODEL_STRUCTURE_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "learning_strategy": learning_strategy,
        "max_K": MAX_K,
        "max_samples": MAX_SAMPLES,
        "batch_size": BATCH_SIZE,
        "train_size": TRAIN_SIZE,
        "split_seed": SPLIT_SEED,
        "split_strategy": SPLIT_STRATEGY,
        "val_waveforms": int(len(VAL_INDICES)),
        "seeds": SEEDS,
        "snr_gap_targets": SNR_GAP_TARGETS,
        "homogeneous_targets": HOMOGENEOUS_TARGETS,
        "homogeneous_pool_size": HOMOGENEOUS_POOL_SIZE,
        "noise": NOISE,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    detail_file = None
    writer = None
    if SAVE_SAMPLE_DETAILS:
        detail_file = gzip.open(OUTPUT_DIR / "sample_details.csv.gz", "wt", newline="")
        writer = csv.DictWriter(
            detail_file,
            fieldnames=[
                "dataset",
                "target",
                "seed",
                "idx",
                "true_K",
                "pred_K",
                "pred_score",
                "loss",
                "correct",
                "error_K",
                "abs_error_K",
                "snr_mix",
                "snr_min",
                "snr_max",
                "snr_mean",
                "snr_gap",
                "energy_noisy",
            ],
        )
        writer.writeheader()

    summaries = []
    confusion_matrices = {}
    tasks = [
        ("snr_gap", SNR_GAP_TARGETS),
        ("homogeneous", HOMOGENEOUS_TARGETS),
    ]

    try:
        for kind, targets in tasks:
            for target in targets:
                for seed in SEEDS:
                    print(f"\n=== {kind} | target={target} | seed={seed} ===")
                    try : 
                        dataset = make_dataset(kind, target, seed)
                    except Exception as e:
                        print(f"Failed to create dataset for {kind} target={target} seed={seed}: {e}")
                        continue
                    try:
                        summary, cm = evaluate_one(model, dataset, kind, target, seed, device, writer)
                    except Exception as e:
                        print(f"Failed to evaluate model for {kind} target={target} seed={seed}: {e}")
                        continue

                    summaries.append(summary)
                    save_summary(summaries)
                    confusion_matrices[f"{kind}_target_{target}_seed_{seed}"] = cm
                    np.savez_compressed(OUTPUT_DIR / "confusion_matrices.npz", **confusion_matrices)

                    print(
                        f"accuracy={summary['accuracy']:.4f} "
                        f"macro_f1={summary['macro_f1']:.4f} "
                        f"mae={summary['mae']:.4f}"
                    )
                    del dataset
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
    finally:
        if detail_file is not None:
            detail_file.close()

    print(f"\nDone. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
