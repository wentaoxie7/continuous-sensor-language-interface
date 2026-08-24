import argparse

from pathlib import Path

import torch
import torch.nn as nn

from .config import ContextTrainingConfig
from .context_model import ContextEmbeddingClassifier
from .data import create_dataloaders
from .shuffle_eval import evaluate_with_shuffled_sensor_embeddings
from .utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shuffled-embedding check for a saved context model checkpoint."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("context_embedding_classifier.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    return parser.parse_args()


def run_shuffle_evaluation(checkpoint_path: Path, batch_size: int | None, shuffle_seed: int | None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ContextTrainingConfig.from_dict(checkpoint["config"])

    if batch_size is not None:
        config.batch_size = batch_size
    if shuffle_seed is not None:
        config.shuffle_seed = shuffle_seed

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
    model.load_state_dict(checkpoint["model_state_dict"])
    loaders, _, _ = create_dataloaders(config)

    criterion = nn.CrossEntropyLoss()
    shuffle_loss, shuffle_f1, _, _ = evaluate_with_shuffled_sensor_embeddings(
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
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Shuffle Test Loss: {shuffle_loss:.4f}")
    print(f"Shuffle Test Macro-F1: {shuffle_f1:.4f}")


def main() -> None:
    args = parse_args()
    run_shuffle_evaluation(
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        shuffle_seed=args.shuffle_seed,
    )
