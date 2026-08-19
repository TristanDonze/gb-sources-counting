import torch
from tqdm import tqdm
from config import LAMBDA_MSE, LAMBDA_CE, LAMBDA_ORDINAL, WEIGHT_BY_K, MSE_K_WEIGHT_ALPHA

def train_one_epoch(
    model,
    max_k,
    dataloader,
    criterion,
    optimizer,
    learning_strategy,
    device,
    *,
    lambda_mse=None,
    lambda_ce=None,
    lambda_ordinal=None,
    weight_by_k=None,
    mse_k_weight_alpha=None,
):
    lambda_mse = LAMBDA_MSE if lambda_mse is None else lambda_mse
    lambda_ce = LAMBDA_CE if lambda_ce is None else lambda_ce
    lambda_ordinal = LAMBDA_ORDINAL if lambda_ordinal is None else lambda_ordinal
    weight_by_k = WEIGHT_BY_K if weight_by_k is None else weight_by_k
    mse_k_weight_alpha = MSE_K_WEIGHT_ALPHA if mse_k_weight_alpha is None else mse_k_weight_alpha

    model.train()
    total_loss = 0.0
    for batch_idx, (summed_waveforms, target) in enumerate(tqdm(dataloader)):
        X = torch.as_tensor(summed_waveforms, dtype=torch.float32, device=device)
        logits = model(X)

        if learning_strategy == "mse":
            labels = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(1)
            y = labels / max_k
            out = torch.sigmoid(logits)
            if weight_by_k:
                per_sample_loss = (out - y).pow(2).squeeze(1)
                weights = 1.0 + mse_k_weight_alpha * (labels.squeeze(1) - 1.0) / (max_k - 1.0)
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

            loss = lambda_mse * loss_mse + lambda_ce * loss_ce
        elif learning_strategy == "mse+ce+or":
            logits_mse, logits_ce, logits_ordinal = logits

            labels = torch.as_tensor(target, dtype=torch.long, device=device)

            y_mse = labels.float().unsqueeze(1) / max_k
            out_mse = torch.sigmoid(logits_mse)

            # criterion[0] is MSE, criterion[1] is CE, criterion[2] is Ordinal
            
            # MSE
            if weight_by_k:
                per_sample_loss = (out_mse - y_mse).pow(2).squeeze(1)
                weights = 1.0 + mse_k_weight_alpha * (labels.float() - 1.0) / (max_k - 1.0)
                loss_mse = (weights * per_sample_loss).sum() / weights.sum()
            else:
                loss_mse = criterion[0](out_mse, y_mse)

            # Cross-Entropy
            y_ce = labels - 1
            loss_ce = criterion[1](logits_ce, y_ce)

            # Ordinal
            thresholds = torch.arange(1, max_k, device=device)
            y_ordinal = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
            loss_ordinal = criterion[2](logits_ordinal, y_ordinal)


            loss = (
                lambda_mse * loss_mse
                + lambda_ce * loss_ce
                + lambda_ordinal * loss_ordinal
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
