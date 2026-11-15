"""A small NumPy-only, fully connected feedforward neural network."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.ann.activations import get_activation, sigmoid
from src.ann.architecture import validate_architecture


class NeuralNetwork:
    """Perform forward propagation only; this class deliberately has no training code."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activations: str | Sequence[str] = "relu",
        random_seed: int | None = None,
    ) -> None:
        self.layer_sizes = validate_architecture(layer_sizes)
        hidden_layer_count = len(self.layer_sizes) - 2
        if isinstance(hidden_activations, str):
            activation_names = [hidden_activations] * hidden_layer_count
        else:
            activation_names = list(hidden_activations)
        if len(activation_names) != hidden_layer_count:
            raise ValueError("Provide one hidden activation for each hidden layer.")

        self.hidden_activations = tuple(get_activation(name) for name in activation_names)
        generator = np.random.default_rng(random_seed)
        self.weights = [
            generator.normal(0.0, 1.0 / np.sqrt(input_size), size=(input_size, output_size))
            for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]
        self.biases = [np.zeros(output_size, dtype=float) for output_size in self.layer_sizes[1:]]

    def set_weights_and_biases(
        self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]
    ) -> None:
        """Replace all network weights and biases after validating their shapes."""
        if len(weights) != len(self.weights) or len(biases) != len(self.biases):
            raise ValueError("Weights and biases must contain one array for every network layer.")

        new_weights = [np.asarray(weight, dtype=float) for weight in weights]
        new_biases = [np.asarray(bias, dtype=float) for bias in biases]
        for index, (weight, bias, expected_weight, expected_bias) in enumerate(
            zip(new_weights, new_biases, self.weights, self.biases), start=1
        ):
            if weight.shape != expected_weight.shape:
                raise ValueError(
                    f"Weight matrix {index} has shape {weight.shape}; expected {expected_weight.shape}."
                )
            if bias.shape != expected_bias.shape:
                raise ValueError(
                    f"Bias vector {index} has shape {bias.shape}; expected {expected_bias.shape}."
                )
        self.weights = [weight.copy() for weight in new_weights]
        self.biases = [bias.copy() for bias in new_biases]

    def set_parameters(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> None:
        """Set weights and biases; a concise alias for explicit parameter assignment."""
        self.set_weights_and_biases(weights, biases)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return sigmoid output probabilities for one sample or a batch of samples."""
        inputs = np.asarray(inputs, dtype=float)
        is_single_sample = inputs.ndim == 1
        if is_single_sample:
            inputs = inputs.reshape(1, -1)
        if inputs.ndim != 2 or inputs.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Inputs must have shape (n_samples, {self.layer_sizes[0]}) or ({self.layer_sizes[0]},).")

        activations = inputs
        for weight, bias, activation in zip(self.weights[:-1], self.biases[:-1], self.hidden_activations):
            activations = activation(activations @ weight + bias)
        output = sigmoid(activations @ self.weights[-1] + self.biases[-1])
        return output[0] if is_single_sample else output

    def predict(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using a configurable probability threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Prediction threshold must be between 0 and 1.")
        return (self.forward(inputs) >= threshold).astype(int)
