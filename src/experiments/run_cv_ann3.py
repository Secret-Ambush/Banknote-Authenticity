"""Manually run the isolated five-fold ANN 3 cross-validation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import DEFAULT_RAW_PATH
from src.evaluation.cross_validation import (
    CV_SPLIT_SEED, FOLD_SEEDS, OUTER_SPLIT_SEED, get_ann3_configuration,
    load_outer_development_partition, run_ann3_cross_validation, summarise_cross_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "cv_ann3"
OUTPUT_FILENAMES = ("fold_results.csv", "summary_results.csv", "oof_predictions.csv", "experiment_metadata.json")


def _serialise(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fold_rows(results: list[object]) -> list[dict[str, object]]:
    return [{
        "fold_number": result.fold_number, "training_seed": result.seed,
        "train_size": result.train_size, "validation_size": result.validation_size,
        "train_class_counts": json.dumps(result.train_class_counts),
        "validation_class_counts": json.dumps(result.validation_class_counts),
        "train_accuracy": result.train_accuracy, "best_training_fitness": result.best_training_fitness,
        "validation_accuracy": result.validation_accuracy, "validation_precision": result.validation_precision,
        "validation_recall": result.validation_recall, "validation_f1_score": result.validation_f1_score,
        "fitness_evaluations": result.fitness_evaluations, "execution_time_seconds": result.execution_time_seconds,
    } for result in results]


def save_cross_validation_results(
    results: list[object], summary: dict[str, float], oof: dict[str, object],
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> None:
    """Write CV artefacts only if none of their destination files already exist."""
    output_directory = Path(output_directory)
    paths = [output_directory / name for name in OUTPUT_FILENAMES]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing CV output: {existing[0]}")
    output_directory.mkdir(parents=True, exist_ok=True)
    fold_rows = _fold_rows(results)
    with (output_directory / "fold_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    with (output_directory / "summary_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    with (output_directory / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["development_index", "label", "probability", "prediction"])
        writer.writeheader()
        writer.writerows({"development_index": int(index), "label": int(label), "probability": float(probability), "prediction": int(prediction)}
                         for index, label, probability, prediction in zip(
                             oof["indices"], oof["labels"], oof["probabilities"], oof["predictions"]
                         ))
    configuration = get_ann3_configuration()
    metadata = {
        "model_configuration": {"name": configuration.name, "architecture": list(configuration.architecture),
                                "hidden_activation": configuration.activation, "output_activation": "sigmoid",
                                "swarm_size": configuration.swarm_size, "max_evaluations": configuration.max_evaluations,
                                "alpha": configuration.alpha, "beta": configuration.beta, "gamma": configuration.gamma,
                                "delta": configuration.delta, "epsilon": configuration.epsilon, "informant_count": 3,
                                "classification_threshold": 0.5},
        "dataset": {"path": str(DEFAULT_RAW_PATH), "sha256": hashlib.sha256(DEFAULT_RAW_PATH.read_bytes()).hexdigest(),
                    "observations": 1372, "development_partition": "seeded 70% outer-training partition"},
        "outer_split_seed": OUTER_SPLIT_SEED, "cv_partition_seed": CV_SPLIT_SEED,
        "fold_training_seeds": list(FOLD_SEEDS),
        "folds": [{"fold_number": result.fold_number, "train_size": result.train_size,
                   "validation_size": result.validation_size, "train_class_counts": result.train_class_counts,
                   "validation_class_counts": result.validation_class_counts,
                   "scaler_mean": result.scaler_mean.tolist(), "scaler_std": result.scaler_std.tolist(),
                   "fitness_evaluations": result.fitness_evaluations} for result in results],
        "pooled_oof_metrics": {key: _serialise(value) for key, value in oof.items() if key not in {"indices", "labels", "probabilities", "predictions"}},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "note": "The prior outer 30% test partition is excluded from this CV pipeline and was used by earlier experiments.",
    }
    with (output_directory / "experiment_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_serialise)


def main() -> None:
    """Execute the authorised CV experiment only when invoked as a module/script."""
    parser = argparse.ArgumentParser(description="Run five-fold stratified CV for ANN 3.")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    X_development, y_development = load_outer_development_partition()
    results, oof = run_ann3_cross_validation(X_development, y_development)
    summary = summarise_cross_validation(results)
    save_cross_validation_results(results, summary, oof, args.output_directory)
    print(f"Saved five-fold ANN 3 CV results to {args.output_directory}.")


if __name__ == "__main__":
    main()
