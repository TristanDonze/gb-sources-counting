import torch
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score
from config import LAMBDA_MSE, LAMBDA_CE, LAMBDA_ORDINAL, LAMBDA_PREDICTION_MSE, LAMBDA_PREDICTION_CE, LAMBDA_PREDICTION_ORDINAL, HYBRID_STRATEGIES

def evaluate(
    model,
    max_k,
    dataloader,
    criterion,
    learning_strategy,
    device,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    abs_error_sum = 0.0

    if learning_strategy in HYBRID_STRATEGIES:
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
        if learning_strategy == "mse+ce+or":
            all_preds_multi["ordinal"] = []
            abs_error_sum_multi["ordinal"] = 0.0
            correct_multi["ordinal"] = 0
        total_multi = 0
        total_loss_mse = 0.0
        total_loss_ce = 0.0
        total_loss_ordinal = 0.0

    with torch.no_grad():
        for batch_idx, (summed_waveforms, target) in enumerate(tqdm(dataloader)):
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
            elif learning_strategy in HYBRID_STRATEGIES:
                if learning_strategy == "mse+ce":
                    logits_mse, logits_ce = logits
                    logits_ordinal = None
                else:
                    logits_mse, logits_ce, logits_ordinal = logits

                y_mse = labels.float().unsqueeze(1) / max_k
                y_ce = labels - 1

                out_mse = torch.sigmoid(logits_mse)

                loss_mse = criterion[0](out_mse, y_mse)
                loss_ce = criterion[1](logits_ce, y_ce)
                loss = LAMBDA_MSE * loss_mse + LAMBDA_CE * loss_ce

                total_loss_mse += loss_mse.item()
                total_loss_ce += loss_ce.item()

                if learning_strategy == "mse+ce+or":
                    thresholds = torch.arange(1, max_k, device=device)
                    y_ordinal = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
                    loss_ordinal = criterion[2](logits_ordinal, y_ordinal)
                    loss += LAMBDA_ORDINAL * loss_ordinal
                    total_loss_ordinal += loss_ordinal.item()
                
                mse_score = out_mse.squeeze(1) * max_k
                predicted_mse = torch.round(mse_score).long()
                predicted_mse = predicted_mse.clamp(1, max_k)

                predicted_ce = logits_ce.argmax(dim=1) + 1

                if learning_strategy == "mse+ce+or":
                    ordinal_probs = torch.sigmoid(logits_ordinal) # corresponds to the probability of being greater than each threshold
                    ordinal_score = 1.0 + ordinal_probs.sum(dim=1) # the expected value of the ordinal prediction
                    predicted_ordinal = 1 + (ordinal_probs > 0.5).sum(dim=1)
                    predicted_ordinal = predicted_ordinal.clamp(1, max_k)

                # Combined predictions
                ce_probs = torch.softmax(logits_ce, dim=1)
                class_values = torch.arange(1, max_k + 1, device=device).float()
                ce_score = (ce_probs * class_values.unsqueeze(0)).sum(dim=1)

                if learning_strategy == "mse+ce":
                    final_score = LAMBDA_PREDICTION_MSE * mse_score + LAMBDA_PREDICTION_CE * ce_score
                elif learning_strategy == "mse+ce+or":
                    final_score = (LAMBDA_PREDICTION_MSE * mse_score + 
                                   LAMBDA_PREDICTION_CE * ce_score + 
                                   LAMBDA_PREDICTION_ORDINAL * ordinal_score)

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
            elif learning_strategy in HYBRID_STRATEGIES:
                total_multi += labels.size(0)
                batch_labels = labels.cpu().numpy()
                all_labels_multi.extend(batch_labels)

                predictor_preds = [
                    ("mse", predicted_mse),
                    ("ce", predicted_ce),
                    ("ensemble", pred_combined),
                ]
                if learning_strategy == "mse+ce+or":
                    predictor_preds.append(("ordinal", predicted_ordinal))

                for predictor, pred in predictor_preds:
                    correct_multi[predictor] += (pred == labels).sum().item()
                    abs_error_sum_multi[predictor] += torch.abs(pred - labels).sum().item()
                    all_preds_multi[predictor].extend(pred.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    if learning_strategy in HYBRID_STRATEGIES:
        metrics = {
            "loss": avg_loss,
            "loss_mse": total_loss_mse / len(dataloader),
            "loss_ce": total_loss_ce / len(dataloader),
            "loss_ordinal": (
                total_loss_ordinal / len(dataloader)
                if learning_strategy == "mse+ce+or"
                else None
            ),
            "predictors": {},
        }
        for predictor in all_preds_multi:
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
