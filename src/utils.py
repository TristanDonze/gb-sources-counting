import torch
def save_checkpoint(model, optimizer, scheduler, train_losses, val_losses, val_accs, val_recall, val_f1s, best_val_f1, best_val_f1_epoch, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_recall": val_recall,
        "val_f1s": val_f1s,
        "best_val_f1": best_val_f1,
        "best_val_f1_epoch": best_val_f1_epoch,
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
    best_val_f1 = checkpoint["best_val_f1"]
    best_val_f1_epoch = checkpoint["best_val_f1_epoch"]
    epoch = checkpoint["epoch"]
    return train_losses, val_losses, val_accs, val_recall, val_f1s, best_val_f1, best_val_f1_epoch, epoch