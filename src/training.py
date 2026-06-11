import torch
from config import LAMBDA_MSE, LAMBDA_CE

def train_one_epoch(
    model,
    max_k,
    dataloader,
    criterion,
    optimizer,
    learning_strategy,
    device,
):
    model.train()
    total_loss = 0.0
    for batch_idx, (summed_waveforms, target) in enumerate(dataloader):
        X = torch.as_tensor(summed_waveforms, dtype=torch.float32, device=device)
        logits = model(X)

        if learning_strategy == "mse":
            labels = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(1)
            y = labels / max_k
            out = torch.sigmoid(logits)
            loss = criterion(out, y)
        elif learning_strategy == "cross_entropy":
            labels = torch.as_tensor(target, dtype=torch.long, device=device)
            y = labels - 1
            loss = criterion(logits, y)
        elif learning_strategy == "ordinal":
            labels = torch.as_tensor(target, dtype=torch.long, device=device)
            thresholds = torch.arange(1, max_k, device=device)
            y = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
            loss = criterion(logits, y)
        elif learning_strategy == "mse+ce":
            logits_mse, logits_ce = logits

            labels_mse = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(1)
            y_mse = labels_mse / max_k
            out_mse = torch.sigmoid(logits_mse)

            labels_ce = torch.as_tensor(target, dtype=torch.long, device=device)
            y_ce = labels_ce - 1
            
            # criterion[0] is MSE, criterion[1] is CE
            loss_mse = criterion[0](out_mse, y_mse)
            loss_ce = criterion[1](logits_ce, y_ce)

            loss = LAMBDA_MSE * loss_mse + LAMBDA_CE * loss_ce

        else:
            raise ValueError(f"Invalid learning strategy: {learning_strategy}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
    avg_loss = total_loss / len(dataloader)
    return avg_loss
