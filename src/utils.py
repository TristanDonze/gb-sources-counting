import os
import torch


def save_checkpoint(
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
    path,
    val_energy_matched_losses=None,
    val_energy_matched_accs=None,
    val_energy_matched_recall=None,
    val_energy_matched_f1s=None,
    val_energy_matched_maes=None,
    min_lr_reached_epoch=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_recall": val_recall,
        "val_f1s": val_f1s,
        "val_maes": val_maes,
        "val_energy_matched_losses": val_energy_matched_losses or [],
        "val_energy_matched_accs": val_energy_matched_accs or [],
        "val_energy_matched_recall": val_energy_matched_recall or [],
        "val_energy_matched_f1s": val_energy_matched_f1s or [],
        "val_energy_matched_maes": val_energy_matched_maes or [],
        "best_val_f1": best_val_f1,
        "best_val_f1_epoch": best_val_f1_epoch,
        "min_lr_reached_epoch": min_lr_reached_epoch,
        "epoch": epoch
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    val_accs = checkpoint["val_accs"]
    val_recall = checkpoint["val_recall"]
    val_f1s = checkpoint["val_f1s"]
    val_maes = checkpoint["val_maes"]
    val_energy_matched_losses = checkpoint.get("val_energy_matched_losses", [])
    val_energy_matched_accs = checkpoint.get("val_energy_matched_accs", [])
    val_energy_matched_recall = checkpoint.get("val_energy_matched_recall", [])
    val_energy_matched_f1s = checkpoint.get("val_energy_matched_f1s", [])
    val_energy_matched_maes = checkpoint.get("val_energy_matched_maes", [])
    best_val_f1 = checkpoint["best_val_f1"]
    best_val_f1_epoch = checkpoint["best_val_f1_epoch"]
    min_lr_reached_epoch = checkpoint.get("min_lr_reached_epoch")
    epoch = checkpoint["epoch"]
    return (
        train_losses,
        val_losses,
        val_accs,
        val_recall,
        val_f1s,
        val_maes,
        best_val_f1,
        best_val_f1_epoch,
        epoch,
        val_energy_matched_losses,
        val_energy_matched_accs,
        val_energy_matched_recall,
        val_energy_matched_f1s,
        val_energy_matched_maes,
        min_lr_reached_epoch,
    )
