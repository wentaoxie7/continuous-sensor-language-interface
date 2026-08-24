import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from .config import CLASS_NAMES, ENCODER_OUTPUT_DIM, TrainingConfig
from .data import create_dataloaders
from .engine import evaluate, train_one_epoch
from .models import DirectSensorClassifier
from .utils import count_parameters, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HAR classifier on UCI HAR Dataset.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-subject-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--checkpoint-path", type=str, default="direct_sensor_classifier.pt")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        base_dir=args.base_dir.resolve(),
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_subject_fraction=args.val_subject_fraction,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint_path,
    )


def print_shape_check(model: DirectSensorClassifier, train_loader, device: torch.device) -> None:
    X_batch, _ = next(iter(train_loader))
    X_batch = X_batch.to(device)

    with torch.no_grad():
        features = model.sensor_encoder(X_batch)
        logits = model(X_batch)

    print("\nShape check")
    print(f"Input:             {tuple(X_batch.shape)}")
    print(f"Encoder output:    {tuple(features.shape)}")
    print(f"Classifier output: {tuple(logits.shape)}")


def save_checkpoint(
    model: DirectSensorClassifier,
    config: TrainingConfig,
    best_epoch: int,
    best_val_f1: float,
    test_f1: float,
    train_subject_ids,
    val_subject_ids,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_f1,
        "seed": config.seed,
        "train_subject_ids": train_subject_ids,
        "val_subject_ids": val_subject_ids,
        "architecture": "3-layer 1D CNN",
        "encoder_output_dim": ENCODER_OUTPUT_DIM,
        "class_names": CLASS_NAMES,
        "config": config.to_dict(),
    }
    torch.save(checkpoint, config.checkpoint_path)
    print(f"\nSaved model to {config.checkpoint_path}")


def run_training(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    loaders, train_subject_ids, val_subject_ids = create_dataloaders(config)

    model = DirectSensorClassifier(num_classes=config.num_classes).to(device)
    print("\nModel:")
    print(model)
    print_shape_check(model, loaders["train"], device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_model_state = None

    print("\nStarting training...\n")
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss, val_f1, _, _ = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Macro-F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_model_state)
    model = model.to(device)

    print(f"\nBest validation epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")

    test_loss, test_f1, test_labels, test_predictions = evaluate(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
    )

    print("\n================================")
    print("FINAL TEST RESULT")
    print("================================")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")
    print("\nClassification Report:\n")
    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    save_checkpoint(
        model=model,
        config=config,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        test_f1=test_f1,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run_training(config)
