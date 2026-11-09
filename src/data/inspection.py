"""Inspection and validation helpers for the UCI Banknote dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.download import DEFAULT_RAW_PATH


COLUMN_NAMES = ["variance", "skewness", "curtosis", "entropy", "class"]
FEATURE_COLUMNS = COLUMN_NAMES[:-1]
EXPECTED_ROWS = 1_372


def load_raw_dataset(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load the headerless raw dataset with the coursework column names."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {path}. Run download.py first.")
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate the data requirements specified for Phase 1."""
    if list(data.columns) != COLUMN_NAMES or data.shape[1] != 5:
        raise ValueError(f"Expected exactly these five columns: {COLUMN_NAMES}")
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} records, found {len(data)}.")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in FEATURE_COLUMNS):
        raise ValueError("All four input features must be numerical.")
    if not pd.api.types.is_numeric_dtype(data["class"]):
        raise ValueError("The target column must be numerical.")
    if data.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    numeric_values = data[COLUMN_NAMES].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Dataset contains non-finite numerical values.")
    labels = set(data["class"].unique())
    if labels != {0, 1}:
        raise ValueError(f"Target labels must be 0 and 1; found {sorted(labels)}.")


def inspection_report(data: pd.DataFrame) -> dict[str, Any]:
    """Return the required descriptive inspection information."""
    validate_dataset(data)
    return {
        "dimensions": data.shape,
        "first_five_records": data.head(),
        "last_five_records": data.tail(),
        "column_names": list(data.columns),
        "data_types": data.dtypes,
        "missing_values": data.isna().sum(),
        "duplicate_row_count": int(data.duplicated().sum()),
        "minimum": data.min(),
        "maximum": data.max(),
        "mean": data.mean(),
        "standard_deviation": data.std(),
        "class_distribution": data["class"].value_counts().sort_index(),
        "class_proportions": data["class"].value_counts(normalize=True).sort_index(),
        "unique_target_labels": sorted(data["class"].unique().tolist()),
    }


def print_inspection_report(data: pd.DataFrame) -> None:
    """Print the inspection report in a readable format without changing data."""
    for title, value in inspection_report(data).items():
        print(f"\n{title.replace('_', ' ').title()}:\n{value}")


def main() -> None:
    """Inspect the raw dataset from the command line."""
    parser = argparse.ArgumentParser(description="Inspect the UCI Banknote dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    args = parser.parse_args()
    print_inspection_report(load_raw_dataset(args.input))


if __name__ == "__main__":
    main()
