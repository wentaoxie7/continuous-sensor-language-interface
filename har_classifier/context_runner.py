import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from .config import CLASS_NAMES, ContextTrainingConfig
from .context_model import ContextEmbeddingClassifier
from .data import create_dataloaders
from .engine import evaluate, train_one_epoch
from .shuffle_eval import evaluate_with_shuffled_sensor_embeddings
from .utils import (
    count_parameters,
    count_trainable_parameters,
    get_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the context-embedding HAR model with a frozen SmolLM2 backbone."
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-subject-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--checkpoint-path", type=str, default="context_embedding_classifier.pt")
    parser.add_argument("--llm-model-name", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--sensor-placeholder", type=str, default="<SENSOR>")
    parser.add_argument("--projector-hidden-dim", type=int, default=512)
    parser.add_argument("--llm-dtype", type=str, default="auto")
    parser.add_argument("--max-trainable-params", type=int, default=10_000_000)
    parser.add_argument("--shuffle-seed", type=int, default=43)
    parser.add_argument("--skip-shuffle-check", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ContextTrainingConfig:
    config = ContextTrainingConfig(
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
        llm_model_name=args.llm_model_name,
        sensor_placeholder=args.sensor_placeholder,
        projector_hidden_dim=args.projector_hidden_dim,
        llm_dtype=args.llm_dtype,
        max_trainable_params=args.max_trainable_params,
        run_shuffle_check=not args.skip_shuffle_check,
        shuffle_seed=args.shuffle_seed,
    )
    if args.prompt_template is not None:
        config.prompt_template = args.prompt_template
    return config


def print_shape_check(model: ContextEmbeddingClassifier, train_loader, device: torch.device) -> None:
    X_batch, _ = next(iter(train_loader))
    X_batch = X_batch.to(device)

    with torch.no_grad():
        sensor_embeddings = model.encode_sensor(X_batch)
        inputs_embeds, _ = model.build_inputs_embeds(sensor_embeddings)
        logits = model(x=X_batch)

    print("\nShape check")
    print(f"Input:               {tuple(X_batch.shape)}")
    print(f"Projected sensor:    {tuple(sensor_embeddings.shape)}")
    print(f"Prompted embeds:     {tuple(inputs_embeds.shape)}")
    print(f"Classifier output:   {tuple(logits.shape)}")


def summarize_trainable_parameters(model: ContextEmbeddingClassifier) -> dict[str, int]:
    return {
        "sensor_encoder": count_trainable_parameters(model.sensor_encoder),
        "sensor_projector": count_trainable_parameters(model.sensor_projector),
        "classifier": count_trainable_parameters(model.classifier),
    }


def print_parameter_summary(model: ContextEmbeddingClassifier, max_trainable_params: int) -> None:
    total_params, trainable_params = count_parameters(model)
    trainable_parts = summarize_trainable_parameters(model)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("Trainable breakdown:")
    for name, value in trainable_parts.items():
        print(f"  {name}: {value:,}")

    if trainable_params > max_trainable_params:
        raise ValueError(
            f"Trainable parameter budget exceeded: {trainable_params:,} > {max_trainable_params:,}"
        )


def save_checkpoint(
    model: ContextEmbeddingClassifier,
    config: ContextTrainingConfig,
    best_epoch: int,
    best_val_f1: float,
    test_f1: float,
    shuffled_test_f1: float | None,
    train_subject_ids,
    val_subject_ids,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_f1,
        "shuffled_test_macro_f1": shuffled_test_f1,
        "seed": config.seed,
        "train_subject_ids": train_subject_ids,
        "val_subject_ids": val_subject_ids,
        "architecture": "sensor encoder + projector + frozen SmolLM2 + linear head",
        "class_names": CLASS_NAMES,
        "config": config.to_dict(),
    }
    torch.save(checkpoint, config.checkpoint_path)
    print(f"\nSaved model to {config.checkpoint_path}")


def run_context_training(config: ContextTrainingConfig) -> None:
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    model = ContextEmbeddingClassifier(
        num_classes=config.num_classes,
        llm_model_name=config.llm_model_name,
        prompt_template=config.prompt_template,
        sensor_placeholder=config.sensor_placeholder,
        projector_hidden_dim=config.projector_hidden_dim,
        device=device,
        llm_dtype=config.llm_dtype,
    ).to(device)

    loaders, train_subject_ids, val_subject_ids = create_dataloaders(config)

    print("\nModel:")
    print(model)
    print_shape_check(model, loaders["train"], device)
    print_parameter_summary(model, config.max_trainable_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
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

    shuffled_test_f1 = None
    if config.run_shuffle_check:
        shuffle_loss, shuffled_test_f1, _, _ = evaluate_with_shuffled_sensor_embeddings(
            model=model,
            dataset=loaders["test"].dataset,
            batch_size=config.batch_size,
            criterion=criterion,
            device=device,
            shuffle_seed=config.shuffle_seed,
        )
        print("\n================================")
        print("SHUFFLED SENSOR CHECK")
        print("================================")
        print(f"Shuffle Test Loss: {shuffle_loss:.4f}")
        print(f"Shuffle Test Macro-F1: {shuffled_test_f1:.4f}")

    save_checkpoint(
        model=model,
        config=config,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        test_f1=test_f1,
        shuffled_test_f1=shuffled_test_f1,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run_context_training(config)
