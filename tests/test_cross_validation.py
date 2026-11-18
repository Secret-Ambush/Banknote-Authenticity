"""Fast unit tests for the separate ANN 3 cross-validation extension."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.evaluation import cross_validation as cv
from src.experiments.run_cv_ann3 import save_cross_validation_results


def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat([0, 1], 10)
    features = np.column_stack((labels * 2.0 - 1.0, np.arange(20), np.arange(20) ** 2, np.ones(20)))
    return features, labels


def fake_fold_result(index: int, *, accuracy: float = 0.5) -> cv.FoldResult:
    return cv.FoldResult(
        fold_number=index + 1, seed=4201 + index, train_size=16, validation_size=4,
        train_class_counts={0: 8, 1: 8}, validation_class_counts={0: 2, 1: 2},
        scaler_mean=np.zeros(4), scaler_std=np.ones(4), train_accuracy=accuracy,
        best_training_fitness=accuracy, validation_accuracy=accuracy, validation_precision=accuracy,
        validation_recall=accuracy, validation_f1_score=accuracy, fitness_evaluations=1000,
        execution_time_seconds=float(index + 1), validation_indices=np.arange(index * 4, index * 4 + 4),
        validation_labels=np.array([0, 0, 1, 1]), validation_probabilities=np.array([0.1, 0.1, 0.9, 0.9]),
    )


def test_stratified_folds_are_disjoint_cover_all_rows_and_are_reproducible() -> None:
    _, labels = synthetic_data()
    first = cv.generate_stratified_folds(labels, seed=42)
    second = cv.generate_stratified_folds(labels, seed=42)
    covered = np.concatenate([fold.validation for fold in first])
    assert np.array_equal(np.sort(covered), np.arange(labels.size))
    assert len(np.unique(covered)) == labels.size
    for left, right in zip(first, second):
        assert np.array_equal(left.train, right.train)
        assert np.array_equal(left.validation, right.validation)
        assert not np.intersect1d(left.train, left.validation).size
        assert {label: int(np.sum(labels[left.validation] == label)) for label in (0, 1)} == {0: 2, 1: 2}


def test_fold_local_standardisation_uses_only_training_rows() -> None:
    X, labels = synthetic_data()
    fold = cv.generate_stratified_folds(labels)[0]
    X_train, X_validation, mean, std = cv.standardise_features(X[fold.train], X[fold.validation])
    assert np.allclose(mean, X[fold.train].mean(axis=0))
    assert np.allclose(std, np.where(X[fold.train].std(axis=0) == 0, 1.0, X[fold.train].std(axis=0)))
    assert np.allclose(X_train.mean(axis=0), 0.0)
    assert np.allclose(X_validation, (X[fold.validation] - mean) / std)


def test_ann3_configuration_is_exact_selected_model() -> None:
    configuration = cv.get_ann3_configuration()
    assert configuration.name == "ANN 3"
    assert configuration.architecture == (4, 16, 1)
    assert configuration.activation == "relu"
    assert (configuration.swarm_size, configuration.max_evaluations) == (20, 1000)
    assert (configuration.alpha, configuration.beta, configuration.gamma, configuration.delta, configuration.epsilon) == (0.5, 1.0, 1.0, 1.0, 1.0)


def test_cv_passes_training_partition_only_and_exact_pso_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    X, labels = synthetic_data()
    trainer_instances: list[FakeTrainer] = []

    class FakeNetwork:
        def __init__(self, layer_sizes, hidden_activations, random_seed):
            self.layer_sizes = tuple(layer_sizes)
            self.hidden_activations = hidden_activations
            self.random_seed = random_seed

        def forward(self, values):
            values = np.asarray(values)
            return np.where(values[:, 0] >= 0, 0.9, 0.1).reshape(-1, 1)

    class FakeTrainer:
        def __init__(self, network, X_train, y_train, **settings):
            self.network = network
            self.X_train = np.asarray(X_train).copy()
            self.y_train = np.asarray(y_train).copy()
            self.settings = settings
            self.optimiser = None
            trainer_instances.append(self)

        def train(self):
            self.optimiser = type("Optimiser", (), {"fitness_evaluations": self.settings["max_evaluations"]})()
            return np.zeros(1), 0.75

    monkeypatch.setattr(cv, "NeuralNetwork", FakeNetwork)
    monkeypatch.setattr(cv, "PSOANNTrainer", FakeTrainer)
    results, oof = cv.run_ann3_cross_validation(X, labels)
    folds = cv.generate_stratified_folds(labels)
    assert len(results) == len(trainer_instances) == 5
    for result, trainer, fold in zip(results, trainer_instances, folds):
        expected_scaled, _, _, _ = cv.standardise_features(X[fold.train], X[fold.validation])
        assert np.allclose(trainer.X_train, expected_scaled)
        assert not np.shares_memory(trainer.X_train, X[fold.validation])
        assert trainer.settings["informant_count"] == 3
        assert trainer.settings["swarm_size"] == 20
        assert trainer.settings["max_evaluations"] == 1000
        assert result.fitness_evaluations == 1000
    assert np.array_equal(oof["indices"], np.arange(len(labels)))
    assert oof["confusion_matrix"] == [[10, 0], [0, 10]]


def test_pooled_oof_coverage_and_summary_statistics() -> None:
    results = [fake_fold_result(index, accuracy=0.4 + index * 0.1) for index in range(5)]
    oof = cv.pooled_oof_metrics(results, development_size=20)
    summary = cv.summarise_cross_validation(results)
    assert np.array_equal(oof["indices"], np.arange(20))
    assert oof["accuracy"] == 1.0  # Pooled predictions intentionally differ from fold summaries.
    assert np.isclose(summary["mean_validation_accuracy"], 0.6)
    assert np.isclose(summary["std_validation_accuracy"], np.std([0.4, 0.5, 0.6, 0.7, 0.8], ddof=1))
    assert summary["min_validation_accuracy"] == 0.4
    assert summary["max_validation_accuracy"] == 0.8


def test_cv_writer_refuses_overwrite_and_existing_results_are_unchanged(tmp_path) -> None:
    existing_result = "experiments/results/architecture_results/individual_results.csv"
    before = hashlib.sha256(open(existing_result, "rb").read()).hexdigest()
    results = [fake_fold_result(index) for index in range(5)]
    oof = cv.pooled_oof_metrics(results, development_size=20)
    output = tmp_path / "cv_ann3"
    save_cross_validation_results(results, cv.summarise_cross_validation(results), oof, output)
    with pytest.raises(FileExistsError):
        save_cross_validation_results(results, cv.summarise_cross_validation(results), oof, output)
    after = hashlib.sha256(open(existing_result, "rb").read()).hexdigest()
    assert before == after
