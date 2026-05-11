import torch
from sklearn.metrics import recall_score, f1_score


def evaluate(model, max_k, dataloader, criterion, learning_strategy, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    abs_error_sum = 0.0
    with torch.no_grad():
        for batch_idx, (summed_waveforms, target) in enumerate(dataloader):
            X = torch.as_tensor(summed_waveforms, dtype=torch.float32, device=device)
            labels = torch.as_tensor(target, dtype=torch.long, device=device)

            logits = model(X)

            if learning_strategy == "mse":
                y = labels.float().unsqueeze(1) / max_k
                out = torch.sigmoid(logits)
                loss = criterion(out, y)
                predicted = torch.round(out.squeeze(1) * max_k).long()
                predicted = predicted.clamp(1, max_k)
            elif learning_strategy == "cross_entropy":
                y = labels - 1
                loss = criterion(logits, y)
                predicted = logits.argmax(dim=1) + 1
            elif learning_strategy == "ordinal":
                thresholds = torch.arange(1, max_k, device=device)
                y = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
                loss = criterion(logits, y)
                predicted = 1 + (torch.sigmoid(logits) > 0.5).sum(dim=1)
                predicted = predicted.clamp(1, max_k)
            else:
                raise ValueError(f"Invalid learning strategy: {learning_strategy}")

            loss_value = loss.item()
            total_loss += loss_value

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_preds = predicted.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            abs_error_sum += torch.abs(predicted - labels).sum().item()

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    avg_loss = total_loss / len(dataloader)
    acc = correct / total

    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mae = abs_error_sum / total

    return avg_loss, acc, recall, f1, mae
