"""Leakage-safe, reproducible five-fold validation for the selected ANN 3 model."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.ann.neural_network import NeuralNetwork
from src.data.inspection import FEATURE_COLUMNS, load_raw_dataset, validate_dataset
from src.data.preprocessing import standardise_features, stratified_train_test_split
from src.evaluation.trainer import PSOANNTrainer
from src.experiments.configurations import ExperimentConfiguration, get_configurations
from src.experiments.metrics import binary_metrics


OUTER_SPLIT_SEED = 42
CV_SPLIT_SEED = 42
FOLD_SEEDS = (4201, 4202, 4203, 4204, 4205)
N_SPLITS = 5


@dataclass(frozen=True)
class FoldIndices:
    """Indices of one development-data training/validation partition."""

    train: np.ndarray
    validation: np.ndarray


@dataclass
class FoldResult:
    """Measurements and audit information for one fitted cross-validation fold."""

    fold_number: int
    seed: int
    train_size: int
    validation_size: int
    train_class_counts: dict[int, int]
    validation_class_counts: dict[int, int]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    train_accuracy: float
    best_training_fitness: float
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    validation_f1_score: float
    fitness_evaluations: int
    execution_time_seconds: float
    validation_indices: np.ndarray
    validation_labels: np.ndarray
    validation_probabilities: np.ndarray


def get_ann3_configuration() -> ExperimentConfiguration:
    """Return the unchanged predefined ANN 3 experiment setting.

    The existing configuration leaves ``informant_count`` as ``None``; the PSO's
    documented default resolves that to three for a swarm of 20.  The CV runner
    passes 3 explicitly so the selected protocol is recorded unambiguously.
    """
    return next(
        configuration
        for configuration in get_configurations(["architecture"])
        if configuration.name == "ANN 3"
    )


def generate_stratified_folds(
    labels: np.ndarray, *, n_splits: int = N_SPLITS, seed: int = CV_SPLIT_SEED
) -> tuple[FoldIndices, ...]:
    """Return deterministic, disjoint stratified folds without scikit-learn."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    generator = np.random.default_rng(seed)
    validation_parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size < n_splits:
            raise ValueError("Every class must contain at least n_splits observations.")
        generator.shuffle(class_indices)
        for fold_number, part in enumerate(np.array_split(class_indices, n_splits)):
            validation_parts[fold_number].append(part)

    all_indices = np.arange(labels.size)
    folds: list[FoldIndices] = []
    for parts in validation_parts:
        validation = np.sort(np.concatenate(parts))
        train = np.setdiff1d(all_indices, validation, assume_unique=True)
        folds.append(FoldIndices(train=train, validation=validation))
    return tuple(folds)


def load_outer_development_partition(
    *, outer_split_seed: int = OUTER_SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the raw 70% development partition used by the original study."""
    data = load_raw_dataset()
    validate_dataset(data)
    X_train, _, y_train, _ = stratified_train_test_split(data, random_seed=outer_split_seed)
    return X_train, y_train


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def run_ann3_cross_validation(
    X_development: np.ndarray,
    y_development: np.ndarray,
    *,
    cv_split_seed: int = CV_SPLIT_SEED,
    fold_seeds: tuple[int, ...] = FOLD_SEEDS,
) -> tuple[list[FoldResult], dict[str, object]]:
    """Fit ANN 3 once per fold and return fold and pooled OOF measurements.

    ``X_development`` must be raw (not previously standardised) features from
    the outer training partition. Validation data are only forwarded after
    ``PSOANNTrainer.train`` has completed.
    """
    X_development = np.asarray(X_development, dtype=float)
    y_development = np.asarray(y_development)
    if X_development.ndim != 2 or X_development.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("X_development must have shape (n_samples, 4).")
    if y_development.ndim != 1 or y_development.shape[0] != X_development.shape[0]:
        raise ValueError("y_development must match X_development's sample count.")
    folds = generate_stratified_folds(y_development, seed=cv_split_seed)
    if len(fold_seeds) != len(folds) or len(set(fold_seeds)) != len(fold_seeds):
        raise ValueError("Provide one unique training seed for every fold.")

    configuration = get_ann3_configuration()
    results: list[FoldResult] = []
    for fold_number, (fold, seed) in enumerate(zip(folds, fold_seeds), start=1):
        X_train_raw, y_train = X_development[fold.train], y_development[fold.train]
        X_validation_raw, y_validation = X_development[fold.validation], y_development[fold.validation]
        X_train, X_validation, mean, std = standardise_features(X_train_raw, X_validation_raw)
        network = NeuralNetwork(configuration.architecture, hidden_activations=configuration.activation, random_seed=seed)
        trainer = PSOANNTrainer(
            network, X_train, y_train, swarm_size=configuration.swarm_size,
            max_evaluations=configuration.max_evaluations, alpha=configuration.alpha,
            beta=configuration.beta, gamma=configuration.gamma, delta=configuration.delta,
            epsilon=configuration.epsilon, informant_count=3, seed=seed,
        )
        start_time = perf_counter()
        _, best_fitness = trainer.train()
        execution_time = perf_counter() - start_time
        train_accuracy = binary_metrics(network.forward(X_train), y_train)["accuracy"]
        validation_probabilities = np.asarray(network.forward(X_validation), dtype=float).reshape(-1)
        validation_metrics = binary_metrics(validation_probabilities, y_validation)
        results.append(FoldResult(
            fold_number=fold_number, seed=seed, train_size=len(fold.train), validation_size=len(fold.validation),
            train_class_counts=_class_counts(y_train), validation_class_counts=_class_counts(y_validation),
            scaler_mean=mean, scaler_std=std, train_accuracy=train_accuracy, best_training_fitness=best_fitness,
            validation_accuracy=validation_metrics["accuracy"], validation_precision=validation_metrics["precision"],
            validation_recall=validation_metrics["recall"], validation_f1_score=validation_metrics["f1_score"],
            fitness_evaluations=trainer.optimiser.fitness_evaluations if trainer.optimiser else 0,
            execution_time_seconds=execution_time, validation_indices=fold.validation.copy(),
            validation_labels=y_validation.copy(), validation_probabilities=validation_probabilities.copy(),
        ))
    return results, pooled_oof_metrics(results, y_development.size)


def pooled_oof_metrics(results: list[FoldResult], development_size: int) -> dict[str, object]:
    """Calculate pooled, one-prediction-per-development-observation OOF metrics."""
    if not results:
        raise ValueError("At least one fold result is required.")
    indices = np.concatenate([result.validation_indices for result in results])
    if indices.size != development_size or np.unique(indices).size != development_size:
        raise ValueError("OOF validation indices must cover every development observation exactly once.")
    order = np.argsort(indices)
    labels = np.concatenate([result.validation_labels for result in results])[order]
    probabilities = np.concatenate([result.validation_probabilities for result in results])[order]
    metrics = binary_metrics(probabilities, labels)
    predictions = (probabilities >= 0.5).astype(int)
    true_negative = int(np.sum((predictions == 0) & (labels == 0)))
    false_positive = int(np.sum((predictions == 1) & (labels == 0)))
    false_negative = int(np.sum((predictions == 0) & (labels == 1)))
    true_positive = int(np.sum((predictions == 1) & (labels == 1)))
    return metrics | {
        "indices": indices[order], "labels": labels, "probabilities": probabilities,
        "predictions": predictions,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def summarise_cross_validation(results: list[FoldResult]) -> dict[str, float]:
    """Return unweighted fold means, sample SDs, minima and maxima."""
    if not results:
        raise ValueError("At least one fold result is required.")
    fields = {
        "validation_accuracy": "validation_accuracy", "validation_precision": "validation_precision",
        "validation_recall": "validation_recall", "validation_f1_score": "validation_f1_score",
        "train_accuracy": "train_accuracy", "best_training_fitness": "best_training_fitness",
        "execution_time_seconds": "execution_time_seconds",
    }
    summary: dict[str, float] = {}
    for output_name, attribute in fields.items():
        values = np.asarray([float(getattr(result, attribute)) for result in results], dtype=float)
        summary[f"mean_{output_name}"] = float(values.mean())
        summary[f"std_{output_name}"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[f"min_{output_name}"] = float(values.min())
        summary[f"max_{output_name}"] = float(values.max())
    return summary
