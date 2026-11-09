"""Tests for the Phase 2 NumPy feedforward ANN."""

from __future__ import annotations

import numpy as np
import pytest

from src.ann.activations import get_activation, relu, sigmoid, tanh
from src.ann.architecture import ARCHITECTURES, get_architecture, validate_architecture
from src.ann.neural_network import NeuralNetwork
from src.ann.parameter_manager import (
    get_parameter_vector,
    restore_parameters,
    set_parameter_vector,
    total_parameter_count,
)
from src.data.preprocessing import load_prepared_dataset


def test_activation_functions_and_lookup() -> None:
    values = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    sigmoid_values = sigmoid(values)
    assert np.all(np.isfinite(sigmoid_values))
    assert np.allclose(sigmoid_values[1:4], [0.26894142, 0.5, 0.73105858])
    assert np.array_equal(relu(values), [0.0, 0.0, 0.0, 1.0, 1000.0])
    assert np.allclose(tanh(values[1:4]), [-0.76159416, 0.0, 0.76159416])
    assert get_activation("relu") is relu
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("softmax")


def test_named_architectures_and_parameter_counts() -> None:
    expected_counts = {"ANN 1": 25, "ANN 2": 49, "ANN 3": 97, "ANN 4": 81, "ANN 5": 225}
    for name, layer_sizes in ARCHITECTURES.items():
        network = NeuralNetwork(get_architecture(name), random_seed=1)
        assert network.layer_sizes == layer_sizes
        assert total_parameter_count(network) == expected_counts[name]


def test_architecture_validation() -> None:
    assert validate_architecture([4, 2, 1]) == (4, 2, 1)
    with pytest.raises(ValueError, match="four input"):
        validate_architecture([3, 2, 1])
    with pytest.raises(ValueError, match="one output"):
        validate_architecture([4, 2, 2])


def test_single_and_batch_forward_propagation() -> None:
    network = NeuralNetwork([4, 2, 1], hidden_activations="tanh", random_seed=7)
    sample = np.array([0.1, -0.2, 0.3, -0.4])
    single_output = network.forward(sample)
    batch_output = network.forward(np.vstack([sample, sample]))
    assert single_output.shape == (1,)
    assert batch_output.shape == (2, 1)
    assert np.all((0.0 <= batch_output) & (batch_output <= 1.0))
    assert network.predict(sample).shape == (1,)
    assert network.predict(np.vstack([sample, sample])).shape == (2, 1)


def test_weight_and_bias_assignment() -> None:
    network = NeuralNetwork([4, 2, 1], random_seed=1)
    weights = [np.ones((4, 2)), np.ones((2, 1))]
    biases = [np.zeros(2), np.array([0.5])]
    network.set_weights_and_biases(weights, biases)
    assert np.array_equal(network.weights[0], weights[0])
    assert np.array_equal(network.biases[1], biases[1])
    with pytest.raises(ValueError, match="Weight matrix"):
        network.set_weights_and_biases([np.ones((3, 2)), weights[1]], biases)


def test_parameter_flattening_and_restoration() -> None:
    network = NeuralNetwork([4, 8, 4, 1], random_seed=3)
    original = get_parameter_vector(network)
    weights, biases = restore_parameters(network, original)
    assert np.array_equal(original, np.concatenate([item.ravel() for pair in zip(weights, biases) for item in pair]))
    set_parameter_vector(network, original + 1.0)
    assert np.array_equal(get_parameter_vector(network), original + 1.0)
    with pytest.raises(ValueError, match="expected"):
        set_parameter_vector(network, original[:-1])


def test_invalid_input_dimensions_and_reproducible_initialisation() -> None:
    first = NeuralNetwork([4, 4, 1], random_seed=42)
    second = NeuralNetwork([4, 4, 1], random_seed=42)
    assert all(np.array_equal(left, right) for left, right in zip(first.weights, second.weights))
    with pytest.raises(ValueError, match="Inputs must have shape"):
        first.forward(np.ones(3))
    with pytest.raises(ValueError, match="Inputs must have shape"):
        first.forward(np.ones((2, 3)))


def test_processed_dataset_integration() -> None:
    data = load_prepared_dataset()
    network = NeuralNetwork([4, 8, 1], random_seed=42)
    assert network.predict(data["X_train"]).shape == (data["X_train"].shape[0], 1)
    assert network.predict(data["X_test"]).shape == (data["X_test"].shape[0], 1)
