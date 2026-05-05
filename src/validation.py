import torch
from sklearn.metrics import recall_score, f1_score

from src.aim_instance import track_metric


def evaluate(model, dataloader, criterion, device):
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
            y = torch.as_tensor(target, dtype=torch.long, device=device)

            logits = model(X)
            loss = criterion(logits, y)
            loss_value = loss.item()
            total_loss += loss_value

            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            batch_preds = predicted.cpu().numpy()
            batch_labels = y.cpu().numpy()
            abs_error_sum += torch.abs(predicted - y).sum().item()

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    avg_loss = total_loss / len(dataloader)
    acc = correct / total

    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mae = abs_error_sum / total

    return avg_loss, acc, recall, f1, mae
