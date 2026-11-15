"""Binary classification metrics and experiment summary statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.evaluation.fitness import binary_classification_fitness


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calculate threshold-0.5 binary metrics with safe zero-denominator handling."""
    accuracy = binary_classification_fitness(probabilities, labels)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    true_positive = np.sum((predictions == 1) & (labels == 1))
    false_positive = np.sum((predictions == 1) & (labels == 0))
    false_negative = np.sum((predictions == 0) & (labels == 1))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1_score": float(f1_score)}


def summarise_runs(run_results: Sequence[dict[str, object]]) -> dict[str, float]:
    """Return requested means and sample standard deviations for one configuration."""
    if not run_results:
        raise ValueError("At least one run result is required for a summary.")

    def mean_and_std(name: str) -> tuple[float, float]:
        values = np.asarray([float(result[name]) for result in run_results], dtype=float)
        return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0

    train_mean, train_std = mean_and_std("train_accuracy")
    test_mean, test_std = mean_and_std("test_accuracy")
    precision_mean, _ = mean_and_std("test_precision")
    recall_mean, _ = mean_and_std("test_recall")
    f1_mean, _ = mean_and_std("test_f1_score")
    time_mean, _ = mean_and_std("execution_time_seconds")
    return {
        "mean_train_accuracy": train_mean,
        "std_train_accuracy": train_std,
        "mean_test_accuracy": test_mean,
        "std_test_accuracy": test_std,
        "mean_test_precision": precision_mean,
        "mean_test_recall": recall_mean,
        "mean_test_f1_score": f1_mean,
        "mean_execution_time_seconds": time_mean,
    }
