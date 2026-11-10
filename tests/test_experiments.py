"""Software tests for the Phase 5 experiment framework."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.ann.neural_network import NeuralNetwork
from src.evaluation.trainer import PSOANNTrainer
from src.experiments.configurations import ExperimentConfiguration, get_configurations
from src.experiments.metrics import binary_metrics, summarise_runs
from src.experiments.runner import (
    INDIVIDUAL_FIELDS,
    SUMMARY_FIELDS,
    ExperimentRunner,
    generate_run_seeds,
    write_results,
)


def small_configuration() -> ExperimentConfiguration:
    """Return a fast configuration used only by software tests."""
    return ExperimentConfiguration("test", "Small", (4, 2, 1), "relu", swarm_size=2, max_evaluations=2)


def test_required_configurations_and_ann_choices() -> None:
    configurations = get_configurations()
    assert len(configurations) == 13
    assert [item.architecture for item in configurations if item.group == "architecture"] == [
        (4, 4, 1), (4, 8, 1), (4, 16, 1), (4, 8, 4, 1), (4, 16, 8, 1)
    ]
    assert {item.activation for item in configurations if item.group == "activation"} == {"sigmoid", "relu", "tanh"}
    budget_configurations = [item for item in configurations if item.group == "budget"]
    assert [(item.swarm_size, item.max_evaluations) for item in budget_configurations] == [(10, 500), (50, 500)]


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="whole number"):
        ExperimentConfiguration("test", "invalid", (4, 2, 1), "relu", swarm_size=3, max_evaluations=4)


def test_trainer_forwards_pso_coefficients() -> None:
    trainer = PSOANNTrainer(
        NeuralNetwork([4, 2, 1], random_seed=1), np.zeros((2, 4)), np.zeros(2),
        swarm_size=2, max_evaluations=2, alpha=0.4, beta=1.2, gamma=0.7, delta=0.9, epsilon=0.8,
        informant_count=2, seed=2,
    )
    trainer.train()
    assert trainer.optimiser is not None
    assert (trainer.optimiser.alpha, trainer.optimiser.beta, trainer.optimiser.gamma) == (0.4, 1.2, 0.7)
    assert (trainer.optimiser.delta, trainer.optimiser.epsilon, trainer.optimiser.informant_count) == (0.9, 0.8, 2)


def test_binary_metrics_and_zero_denominators() -> None:
    metrics = binary_metrics(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 0, 1, 0]))
    assert metrics == {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1_score": 0.5}
    zeros = binary_metrics(np.array([0.1, 0.2]), np.array([0, 0]))
    assert zeros["accuracy"] == 1.0
    assert zeros["precision"] == zeros["recall"] == zeros["f1_score"] == 0.0


def test_summary_uses_sample_standard_deviation() -> None:
    rows = [
        {"train_accuracy": 0.5, "test_accuracy": 0.5, "test_precision": 0.5, "test_recall": 0.5, "test_f1_score": 0.5, "execution_time_seconds": 1.0},
        {"train_accuracy": 1.0, "test_accuracy": 1.0, "test_precision": 1.0, "test_recall": 1.0, "test_f1_score": 1.0, "execution_time_seconds": 3.0},
    ]
    summary = summarise_runs(rows)
    assert summary["mean_train_accuracy"] == 0.75
    assert np.isclose(summary["std_train_accuracy"], np.sqrt(0.125))
    assert summary["mean_execution_time_seconds"] == 2.0


def test_seed_generation_is_deterministic_and_distinct() -> None:
    assert generate_run_seeds(42, 3) == [42, 43, 44]
    assert generate_run_seeds(42, 3) == generate_run_seeds(42, 3)


def test_runner_uses_fresh_training_only_instances() -> None:
    runner = ExperimentRunner(
        np.zeros((2, 4)), np.zeros(2), np.zeros((2, 4)), np.ones(2), base_seed=10
    )
    results, _ = runner.run_configuration(small_configuration(), runs=2)
    assert [result["random_seed"] for result in results] == [10, 11]
    assert all(result["fitness_evaluations"] == 2 for result in results)
    assert all(result["best_training_fitness"] == result["train_accuracy"] for result in results)


def test_csv_output_structure_and_safe_overwrite(tmp_path) -> None:
    individual = [{field: "value" for field in INDIVIDUAL_FIELDS}]
    summary = [{field: "value" for field in SUMMARY_FIELDS}]
    write_results(individual, summary, tmp_path)
    with (tmp_path / "individual_results.csv").open(newline="") as file:
        assert csv.DictReader(file).fieldnames == INDIVIDUAL_FIELDS
    with pytest.raises(FileExistsError, match="already exist"):
        write_results(individual, summary, tmp_path)
