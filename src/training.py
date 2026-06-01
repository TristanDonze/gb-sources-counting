import torch


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
        else:
            raise ValueError(f"Invalid learning strategy: {learning_strategy}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
    avg_loss = total_loss / len(dataloader)
    return avg_loss
