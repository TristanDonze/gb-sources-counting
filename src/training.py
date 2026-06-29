import torch
from config import LAMBDA_MSE, LAMBDA_CE, LAMBDA_ORDINAL, MSE_K_WEIGHT_ALPHA

def train_one_epoch(
    model,
    max_k,
    dataloader,
    criterion,
    optimizer,
    learning_strategy,
    weight_by_K,
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
            if weight_by_K:
                per_sample_loss = (out - y).pow(2).squeeze(1)
                weights = 1.0 + MSE_K_WEIGHT_ALPHA * (labels.squeeze(1) - 1.0) / (max_k - 1.0)
                loss = (weights * per_sample_loss).sum() / weights.sum()
            else:
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
        elif learning_strategy == "mse+ce+or":
            logits_mse, logits_ce, logits_ordinal = logits

            labels = torch.as_tensor(target, dtype=torch.long, device=device)

            y_mse = labels.float().unsqueeze(1) / max_k
            out_mse = torch.sigmoid(logits_mse)

            y_ce = labels - 1

            thresholds = torch.arange(1, max_k, device=device)
            y_ordinal = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()

            # criterion[0] is MSE, criterion[1] is CE, criterion[2] is Ordinal
            loss_mse = criterion[0](out_mse, y_mse)
            loss_ce = criterion[1](logits_ce, y_ce)
            loss_ordinal = criterion[2](logits_ordinal, y_ordinal)

            loss = (
                LAMBDA_MSE * loss_mse
                + LAMBDA_CE * loss_ce
                + LAMBDA_ORDINAL * loss_ordinal
            )

        else:
            raise ValueError(f"Invalid learning strategy: {learning_strategy}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
    avg_loss = total_loss / len(dataloader)
    return avg_loss
