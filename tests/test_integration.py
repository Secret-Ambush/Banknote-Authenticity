"""Tests for Phase 4 PSO-to-ANN integration."""

from __future__ import annotations

import numpy as np
import pytest

from src.ann.neural_network import NeuralNetwork
from src.ann.parameter_manager import get_parameter_vector, total_parameter_count
from src.evaluation.fitness import binary_classification_fitness
from src.evaluation.trainer import PSOANNTrainer


@pytest.fixture
def training_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small four-feature binary training set."""
    return np.zeros((4, 4)), np.zeros(4, dtype=int)


def test_fitness_for_correct_and_incorrect_predictions() -> None:
    assert binary_classification_fitness(np.array([0.9, 0.1]), np.array([1, 0])) == 1.0
    assert binary_classification_fitness(np.array([0.1, 0.9]), np.array([1, 0])) == 0.0


def test_fitness_threshold_is_inclusive_at_half() -> None:
    assert binary_classification_fitness(np.array([[0.5], [0.49]]), np.array([1, 0])) == 1.0


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        (np.array([[0.3, 0.4]]), np.array([0]), "Probabilities"),
        (np.array([0.3]), np.array([0, 1]), "same number"),
        (np.array([0.3]), np.array([2]), "binary"),
        (np.array([np.nan]), np.array([0]), "finite"),
    ],
)
def test_invalid_fitness_inputs_are_rejected(
    probabilities: np.ndarray, labels: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        binary_classification_fitness(probabilities, labels)


def test_particle_parameters_are_assigned_and_forward_propagation_works(
    training_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X_train, y_train = training_data
    network = NeuralNetwork([4, 2, 1], random_seed=1)
    trainer = PSOANNTrainer(network, X_train, y_train, swarm_size=2, max_evaluations=2, seed=2)
    parameters = np.linspace(-0.5, 0.5, total_parameter_count(network))
    fitness = trainer.evaluate_parameters(parameters)
    probabilities = network.forward(X_train)
    assert np.array_equal(get_parameter_vector(network), parameters)
    assert probabilities.shape == (4, 1)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert 0.0 <= fitness <= 1.0


def test_pso_optimises_and_restores_best_ann_parameters(
    training_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X_train, y_train = training_data
    network = NeuralNetwork([4, 2, 1], random_seed=3)
    initial_parameters = get_parameter_vector(network).copy()
    trainer = PSOANNTrainer(
        network,
        X_train,
        y_train,
        swarm_size=4,
        max_evaluations=8,
        seed=4,
        position_bounds=(-2.0, 2.0),
    )
    best_parameters, best_fitness = trainer.train()
    assert best_parameters.shape == (total_parameter_count(network),)
    assert 0.0 <= best_fitness <= 1.0
    assert np.array_equal(get_parameter_vector(network), best_parameters)
    assert not np.array_equal(initial_parameters, best_parameters)


def test_training_is_reproducible_with_the_same_seed(
    training_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X_train, y_train = training_data
    first = PSOANNTrainer(
        NeuralNetwork([4, 2, 1], random_seed=5), X_train, y_train, swarm_size=4, max_evaluations=8, seed=6
    )
    second = PSOANNTrainer(
        NeuralNetwork([4, 2, 1], random_seed=5), X_train, y_train, swarm_size=4, max_evaluations=8, seed=6
    )
    first_parameters, first_fitness = first.train()
    second_parameters, second_fitness = second.train()
    assert np.array_equal(first_parameters, second_parameters)
    assert first_fitness == second_fitness


def test_only_training_data_is_stored_for_optimisation(
    training_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X_train, y_train = training_data
    unused_test_features = np.full((2, 4), 999.0)
    trainer = PSOANNTrainer(
        NeuralNetwork([4, 2, 1], random_seed=7), X_train, y_train, swarm_size=2, max_evaluations=2, seed=8
    )
    trainer.train()
    assert not hasattr(trainer, "X_test")
    assert not np.shares_memory(trainer.X_train, unused_test_features)
