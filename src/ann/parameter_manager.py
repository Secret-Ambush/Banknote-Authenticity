"""Convert ANN weights and biases to and from PSO-friendly parameter vectors."""

from __future__ import annotations

import numpy as np

from src.ann.neural_network import NeuralNetwork


def flatten_parameters(weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    """Flatten weights then biases for each layer into one vector."""
    if len(weights) != len(biases):
        raise ValueError("Weights and biases must have the same number of layers.")
    parts = [part.ravel() for weight, bias in zip(weights, biases) for part in (weight, bias)]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def restore_parameters(network: NeuralNetwork, vector: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Restore a vector into arrays with the network's original parameter shapes."""
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Parameter vector must be one-dimensional.")
    expected_size = total_parameter_count(network)
    if vector.size != expected_size:
        raise ValueError(f"Parameter vector has length {vector.size}; expected {expected_size}.")

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    position = 0
    for weight_template, bias_template in zip(network.weights, network.biases):
        weight_size = weight_template.size
        bias_size = bias_template.size
        weights.append(vector[position : position + weight_size].reshape(weight_template.shape).copy())
        position += weight_size
        biases.append(vector[position : position + bias_size].reshape(bias_template.shape).copy())
        position += bias_size
    return weights, biases


def get_parameter_vector(network: NeuralNetwork) -> np.ndarray:
    """Return the network's current weights and biases as one vector."""
    return flatten_parameters(network.weights, network.biases)


def set_parameter_vector(network: NeuralNetwork, vector: np.ndarray) -> None:
    """Set every network weight and bias from a validated parameter vector."""
    weights, biases = restore_parameters(network, vector)
    network.set_weights_and_biases(weights, biases)


def total_parameter_count(network: NeuralNetwork) -> int:
    """Return the number of scalar weights and biases in a network."""
    return sum(weight.size + bias.size for weight, bias in zip(network.weights, network.biases))
