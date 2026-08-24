from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import SEQUENCE_LENGTH, SIGNAL_NAMES, TrainingConfig


@dataclass(slots=True)
class LoadedSplit:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


@dataclass(slots=True)
class SubjectSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_subject_ids: np.ndarray
    val_subject_ids: np.ndarray


class HARDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def load_har_signals(base_dir: Path, split: str) -> LoadedSplit:
    signal_dir = base_dir / split / "Inertial Signals"
    signals = []

    print(f"\nLoading {split} signals...")
    for signal_name in SIGNAL_NAMES:
        file_path = signal_dir / f"{signal_name}_{split}.txt"
        signal = np.loadtxt(file_path, dtype=np.float32)

        if signal.ndim != 2:
            raise ValueError(f"{file_path} has unexpected shape: {signal.shape}")
        if signal.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"{file_path} should contain {SEQUENCE_LENGTH} time steps per sample, "
                f"but shape is {signal.shape}"
            )

        print(f"{signal_name:20s}: {signal.shape}")
        signals.append(signal)

    num_samples = [signal.shape[0] for signal in signals]
    if len(set(num_samples)) != 1:
        raise ValueError("Signal files have different numbers of samples.")

    X = np.stack(signals, axis=-1)
    y = np.loadtxt(base_dir / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(base_dir / split / f"subject_{split}.txt", dtype=np.int64)

    if len(X) != len(y):
        raise ValueError("X and y have different numbers of samples.")
    if len(X) != len(subjects):
        raise ValueError("X and subject IDs have different numbers of samples.")

    print(f"\n{split} X shape:        {X.shape}")
    print(f"{split} y shape:        {y.shape}")
    print(f"{split} subjects shape: {subjects.shape}")
    print(f"{split} unique subjects: {np.unique(subjects)}")
    print(f"{split} unique labels:   {np.unique(y)}")

    return LoadedSplit(X=X, y=y, subjects=subjects)


def subject_wise_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    val_subject_fraction: float = 0.2,
    seed: int = 42,
) -> SubjectSplit:
    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    num_val_subjects = max(1, round(len(unique_subjects) * val_subject_fraction))
    val_subject_ids = shuffled_subjects[:num_val_subjects]
    train_subject_ids = shuffled_subjects[num_val_subjects:]

    train_mask = np.isin(subjects, train_subject_ids)
    val_mask = np.isin(subjects, val_subject_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print("\nSubject-wise split")
    print(f"Training subjects:   {np.sort(train_subject_ids)}")
    print(f"Validation subjects: {np.sort(val_subject_ids)}")
    print(f"\nX_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"y_val:   {y_val.shape}")

    overlap = np.intersect1d(train_subject_ids, val_subject_ids)
    if len(overlap) != 0:
        raise ValueError(f"Training and validation subjects overlap: {overlap}")

    return SubjectSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
    )


def create_dataloaders(
    config: TrainingConfig,
) -> tuple[dict[str, DataLoader], np.ndarray, np.ndarray]:
    train_full = load_har_signals(config.base_dir, split="train")
    test = load_har_signals(config.base_dir, split="test")

    split = subject_wise_train_val_split(
        X=train_full.X,
        y=train_full.y,
        subjects=train_full.subjects,
        val_subject_fraction=config.val_subject_fraction,
        seed=config.seed,
    )

    datasets = {
        "train": HARDataset(split.X_train, split.y_train),
        "val": HARDataset(split.X_val, split.y_val),
        "test": HARDataset(test.X, test.y),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }

    return loaders, split.train_subject_ids, split.val_subject_ids
