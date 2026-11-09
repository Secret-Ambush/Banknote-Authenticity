"""Activation functions used by the feedforward ANN."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid function element-wise."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def relu(values: np.ndarray) -> np.ndarray:
    """Apply the rectified linear unit function element-wise."""
    return np.maximum(0.0, np.asarray(values, dtype=float))


def tanh(values: np.ndarray) -> np.ndarray:
    """Apply the hyperbolic tangent function element-wise."""
    return np.tanh(np.asarray(values, dtype=float))


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
}


def get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return an activation function by name."""
    try:
        return ACTIVATIONS[name.lower()]
    except (AttributeError, KeyError) as error:
        supported = ", ".join(ACTIVATIONS)
        raise ValueError(f"Unsupported activation '{name}'. Choose one of: {supported}.") from error
