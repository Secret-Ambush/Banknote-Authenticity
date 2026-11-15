"""Leakage-safe stratified preparation for the UCI Banknote dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.inspection import FEATURE_COLUMNS, load_raw_dataset, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "banknote_data.npz"


def stratified_train_test_split(
    data: pd.DataFrame, *, test_fraction: float = 0.30, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a shuffled, reproducible class-wise train/test split without sklearn."""
    validate_dataset(data)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    generator = np.random.default_rng(random_seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    labels = data["class"].to_numpy(dtype=int)
    for label in (0, 1):
        class_indices = np.flatnonzero(labels == label)
        generator.shuffle(class_indices)
        test_count = int(round(len(class_indices) * test_fraction))
        if test_count in (0, len(class_indices)):
            raise ValueError("Each class must have records in both split partitions.")
        test_indices.append(class_indices[:test_count])
        train_indices.append(class_indices[test_count:])

    train_index = np.concatenate(train_indices)
    test_index = np.concatenate(test_indices)
    generator.shuffle(train_index)
    generator.shuffle(test_index)

    features = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    return features[train_index], features[test_index], labels[train_index], labels[test_index]


def standardise_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardise both partitions using training-only population statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / safe_std, (X_test - mean) / safe_std, mean, safe_std


def prepare_dataset(data: pd.DataFrame, *, random_seed: int = 42) -> dict[str, Any]:
    """Split and standardise validated raw data, retaining the original labels."""
    X_train, X_test, y_train, y_test = stratified_train_test_split(data, random_seed=random_seed)
    X_train_scaled, X_test_scaled, mean, std = standardise_features(X_train, X_test)
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "mean": mean,
        "std": std,
        "random_seed": np.array(random_seed),
        "test_fraction": np.array(0.30),
    }


def save_prepared_dataset(prepared_data: dict[str, Any], path: Path = DEFAULT_PROCESSED_PATH) -> Path:
    """Save prepared arrays and preparation metadata to a compressed NPZ archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **prepared_data)
    return path


def load_prepared_dataset(path: Path = DEFAULT_PROCESSED_PATH) -> dict[str, np.ndarray]:
    """Load a saved prepared-data archive into a dictionary."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    """Prepare and save the raw dataset from the command line."""
    parser = argparse.ArgumentParser(description="Prepare the UCI Banknote dataset.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw_data = load_raw_dataset(args.input) if args.input else load_raw_dataset()
    saved_path = save_prepared_dataset(prepare_dataset(raw_data, random_seed=args.seed), args.output)
    print(f"Prepared dataset saved: {saved_path}")


if __name__ == "__main__":
    main()
