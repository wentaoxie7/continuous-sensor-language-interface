import numpy as np
import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def collect_sensor_embeddings(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    all_embeddings = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        X_batch = dataset.X[start:stop].to(device)
        batch_embeddings = model.encode_sensor(X_batch)
        all_embeddings.append(batch_embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def evaluate_with_sensor_embeddings(
    model: torch.nn.Module,
    dataset,
    sensor_embeddings: torch.Tensor,
    batch_size: int,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch_sensor_embeddings = sensor_embeddings[start:stop].to(device)
        y_batch = dataset.y[start:stop].to(device)

        logits = model(sensor_embeddings=batch_sensor_embeddings)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * y_batch.size(0)
        predictions = logits.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    average_loss = total_loss / len(dataset)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    return average_loss, macro_f1, np.array(all_labels), np.array(all_predictions)


@torch.no_grad()
def evaluate_with_shuffled_sensor_embeddings(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    criterion: torch.nn.Module,
    device: torch.device,
    shuffle_seed: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    sensor_embeddings = collect_sensor_embeddings(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )
    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)
    shuffled_embeddings = sensor_embeddings[torch.randperm(len(sensor_embeddings), generator=generator)]
    return evaluate_with_sensor_embeddings(
        model=model,
        dataset=dataset,
        sensor_embeddings=shuffled_embeddings,
        batch_size=batch_size,
        criterion=criterion,
        device=device,
    )
