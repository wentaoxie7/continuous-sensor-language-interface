import numpy as np
import torch
from sklearn.metrics import f1_score


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        predictions = logits.argmax(dim=1)
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    return average_loss, macro_f1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        predictions = logits.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    return average_loss, macro_f1, np.array(all_labels), np.array(all_predictions)
