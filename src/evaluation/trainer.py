"""Connect the existing ANN parameter vector to the existing PSO optimiser."""

from __future__ import annotations

import numpy as np

from src.ann.neural_network import NeuralNetwork
from src.ann.parameter_manager import set_parameter_vector, total_parameter_count
from src.evaluation.fitness import binary_classification_fitness
from src.pso.algorithm import PSO


class PSOANNTrainer:
    """Optimise a network's complete parameter vector using training data only."""

    def __init__(
        self,
        network: NeuralNetwork,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        swarm_size: int = 20,
        max_evaluations: int = 1_000,
        seed: int | None = None,
        position_bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Store the ANN and its training-only data for PSO fitness evaluation."""
        self.network = network
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        if self.X_train.ndim != 2 or self.X_train.shape[1] != network.layer_sizes[0]:
            raise ValueError(f"X_train must have shape (n_samples, {network.layer_sizes[0]}).")
        if self.y_train.ndim != 1 or self.y_train.shape[0] != self.X_train.shape[0]:
            raise ValueError("y_train must be one-dimensional and match X_train's sample count.")
        self.swarm_size = swarm_size
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.position_bounds = position_bounds
        self.optimiser: PSO | None = None

    def evaluate_parameters(self, parameter_vector: np.ndarray) -> float:
        """Assign one particle vector to the ANN and return its training accuracy."""
        set_parameter_vector(self.network, parameter_vector)
        probabilities = self.network.forward(self.X_train)
        return binary_classification_fitness(probabilities, self.y_train)

    def train(self) -> tuple[np.ndarray, float]:
        """Run PSO and restore the best discovered parameters to the ANN."""
        self.optimiser = PSO(
            self.evaluate_parameters,
            dimensions=total_parameter_count(self.network),
            swarm_size=self.swarm_size,
            max_evaluations=self.max_evaluations,
            position_bounds=self.position_bounds,
            seed=self.seed,
        )
        best_parameters, best_fitness = self.optimiser.optimise()
        set_parameter_vector(self.network, best_parameters)
        return best_parameters, best_fitness
