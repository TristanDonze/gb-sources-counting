import torch

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    # add aim to log training loss
    model.train()
    total_loss = 0.0
    for batch_idx, (summed_waveforms, target) in enumerate(dataloader):
        X = torch.tensor(summed_waveforms, dtype=torch.float32).to(device)
        y = torch.tensor(target, dtype=torch.long).to(device)

        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss
