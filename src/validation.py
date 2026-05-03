import torch
from sklearn.metrics import recall_score, f1_score

def evaluate(model, dataloader, criterion, device):
    # add aim to log validation loss, accuracy, recall and f1 score
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (summed_waveforms, target) in enumerate(dataloader):
            X = torch.as_tensor(summed_waveforms, dtype=torch.float32, device=device)
            y = torch.as_tensor(target, dtype=torch.long, device=device)

            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return avg_loss, acc, recall, f1
