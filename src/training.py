import torch
import torch.nn.functional as F

from src.aim_instance import track_metric


def train_one_epoch(model, max_k, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (summed_waveforms, target) in enumerate(dataloader):
        X = torch.as_tensor(summed_waveforms, dtype=torch.float32, device=device)
        labels = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(1)
        y = labels / max_k

        logit = model(X)
        out = torch.sigmoid(logit)

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss
