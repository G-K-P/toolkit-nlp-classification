import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


def eval_epoch_with_confusion_matrix(model, data_loader, device, num_heads=4, label_map=None):
    model.eval()
    predictions = [[] for _ in range(num_heads)]
    truths = [[] for _ in range(num_heads)]

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = [d[f"label{idx + 1}"].to(device) for idx in range(num_heads)]
            for idx in range(num_heads):
                truths[idx] += labels[idx].tolist()

            outputs = model({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            for idx in range(num_heads):
                _, preds_label = torch.max(outputs[idx], dim=1)
                predictions[idx] += preds_label.tolist()

        assert label_map is not None
        # confusion matrix for each layer
        for idx in range(num_heads):
            names = list(label_map[f"level{idx + 1}"].values())
            # print the confusion matrix with DataFrame
            print(pd.DataFrame(confusion_matrix(truths[idx], predictions[idx]), index=names, columns=names))


def eval_epoch(model, data_loader, loss_fn, device, examples, num_heads=4):
    model = model.eval()
    losses = []
    correct_predictions = [0] * num_heads

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = [d[f"label{idx + 1}"].to(device) for idx in range(num_heads)]

            outputs = model({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            # level 1 => full labels
            # level 2 to 4 => lots of empty labels...
            labels_fulfilled = [None, *[labels[idx][labels[idx] != -1] for idx in range(1, num_heads)]]

            for idx in range(num_heads):
                _, preds_label = torch.max(outputs[idx], dim=1)
                if idx == 0:
                    loss_batch = loss_fn(outputs[idx], labels[idx])
                elif labels_fulfilled[idx].shape[0] != 0:
                    loss_batch += loss_fn(outputs[idx][labels[idx] != -1], labels_fulfilled[idx])
                correct_predictions[idx] += torch.sum(preds_label == labels[idx]).item()

            losses.append(loss_batch.item())
    n_examples = [len(examples), *[len(examples[examples[f"label{idx + 1}"] != -1]) for idx in range(1, num_heads)]]
    return [str(round(correct / n_examples[idx], 3)) for idx, correct in enumerate(correct_predictions)], str(
        round(np.mean(losses), 3)), n_examples


class GEvaluator:
    def __init__(self):
        pass
