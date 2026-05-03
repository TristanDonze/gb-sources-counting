import torch

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    # add aim to log training loss
    model.train()
    total_loss = 0.0
    for batch_idx, (summed_waveforms, target) in enumerate(dataloader):
        X = torch.as_tensor(summed_waveforms, dtype=torch.float32, device=device)
        y = torch.as_tensor(target, dtype=torch.long, device=device)

        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss
