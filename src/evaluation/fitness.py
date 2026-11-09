"""Fitness calculations for ANN classification."""

from __future__ import annotations

import numpy as np


def binary_classification_fitness(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return threshold-0.5 binary classification accuracy as a PSO fitness value."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if probabilities.ndim == 2 and probabilities.shape[1] == 1:
        probabilities = probabilities[:, 0]
    if probabilities.ndim != 1:
        raise ValueError("Probabilities must have shape (n_samples,) or (n_samples, 1).")
    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Probabilities and labels must have the same number of samples.")
    if probabilities.size == 0:
        raise ValueError("Probabilities and labels must not be empty.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must contain only binary values 0 and 1.")

    predictions = (probabilities >= 0.5).astype(int)
    return float(np.mean(predictions == labels))
