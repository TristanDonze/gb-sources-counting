import torch
from sklearn.metrics import recall_score, f1_score
from config import LAMBDA_MSE, LAMBDA_CE, LAMBDA_PREDICTION_MSE, LAMBDA_PREDICTION_CE


def evaluate(model, max_k, dataloader, criterion, learning_strategy, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    abs_error_sum = 0.0

    if learning_strategy == "mse+ce":
        all_labels_multi = []
        all_preds_multi = {
            "mse": [],
            "ce": [],
            "ensemble": [],
        }
        abs_error_sum_multi = {
            "mse": 0.0,
            "ce": 0.0,
            "ensemble": 0.0,
        }
        correct_multi = {
            "mse": 0,
            "ce": 0,
            "ensemble": 0,
        }
        total_multi = 0
        total_loss_mse = 0.0
        total_loss_ce = 0.0

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
            elif learning_strategy == "mse+ce":
                logits_mse, logits_ce = logits

                y_mse = labels.float().unsqueeze(1) / max_k
                y_ce = labels - 1

                out_mse = torch.sigmoid(logits_mse)

                loss_mse = criterion[0](out_mse, y_mse)
                loss_ce = criterion[1](logits_ce, y_ce)
                loss = LAMBDA_MSE * loss_mse + LAMBDA_CE * loss_ce
                total_loss_mse += loss_mse.item()
                total_loss_ce += loss_ce.item()
                
                mse_score = out_mse.squeeze(1) * max_k
                predicted_mse = torch.round(mse_score).long()
                predicted_mse = predicted_mse.clamp(1, max_k)

                predicted_ce = logits_ce.argmax(dim=1) + 1

                # Combined predictions
                ce_probs = torch.softmax(logits_ce, dim=1)
                class_values = torch.arange(1, max_k + 1, device=device).float()
                ce_score = (ce_probs * class_values.unsqueeze(0)).sum(dim=1)

                final_score = LAMBDA_PREDICTION_MSE * mse_score + LAMBDA_PREDICTION_CE * ce_score
                pred_combined = torch.round(final_score).long().clamp(1, max_k)

            else:
                raise ValueError(f"Invalid learning strategy: {learning_strategy}")

            loss_value = loss.item()
            total_loss += loss_value

            if learning_strategy in ["mse", "cross_entropy", "ordinal"]:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                batch_preds = predicted.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                abs_error_sum += torch.abs(predicted - labels).sum().item()

                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)
            elif learning_strategy == "mse+ce":
                total_multi += labels.size(0)
                batch_labels = labels.cpu().numpy()
                all_labels_multi.extend(batch_labels)

                for predictor, pred in [
                    ("mse", predicted_mse),
                    ("ce", predicted_ce),
                    ("ensemble", pred_combined),
                ]:
                    correct_multi[predictor] += (pred == labels).sum().item()
                    abs_error_sum_multi[predictor] += torch.abs(pred - labels).sum().item()
                    all_preds_multi[predictor].extend(pred.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    if learning_strategy == "mse+ce":
        metrics = {
            "loss": avg_loss,
            "loss_mse": total_loss_mse / len(dataloader),
            "loss_ce": total_loss_ce / len(dataloader),
            "predictors": {},
        }
        for predictor in ["mse", "ce", "ensemble"]:
            metrics["predictors"][predictor] = {
                "acc": correct_multi[predictor] / total_multi,
                "recall": recall_score(
                    all_labels_multi,
                    all_preds_multi[predictor],
                    average="macro",
                    zero_division=0,
                ),
                "f1": f1_score(
                    all_labels_multi,
                    all_preds_multi[predictor],
                    average="macro",
                    zero_division=0,
                ),
                "mae": abs_error_sum_multi[predictor] / total_multi,
            }
        return metrics

    acc = correct / total

    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mae = abs_error_sum / total

    return avg_loss, acc, recall, f1, mae
