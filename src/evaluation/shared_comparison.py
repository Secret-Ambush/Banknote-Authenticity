"""Reproduce a development-selected Maazuddin model on Riddhi's saved split."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from src.ann.neural_network import NeuralNetwork
from src.ann.parameter_manager import set_parameter_vector, total_parameter_count
from src.data.preprocessing import standardise_features
from src.evaluation.trainer import PSOANNTrainer
from src.experiments.configurations import CONFIGURATIONS, ExperimentConfiguration
from src.experiments.metrics import binary_metrics


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Riddhi" / "ann_results"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_banknotes(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as stream:
        for fields in csv.reader(stream):
            target = fields[-1].strip()
            if target in ("0§", "1§"):
                fields[-1] = target[:-1]
            rows.append([float(value) for value in fields])
    data = np.asarray(rows)
    if data.ndim != 2 or data.shape[1] != 5 or not np.isfinite(data).all():
        raise ValueError("Expected finite banknote rows with four features and a label.")
    if not np.isin(data[:, -1], [0, 1]).all():
        raise ValueError("Expected binary banknote labels.")
    return data


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def shared_split(data: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained = model["retained_original_row_indices"]
    _, first_indices = np.unique(data, axis=0, return_index=True)
    if not np.array_equal(retained, np.sort(first_indices)):
        raise ValueError("Saved retained rows do not match deduplicated source data.")
    development, test = model["development_indices"], model["test_indices"]
    if not np.array_equal(np.sort(np.r_[development, test]), np.arange(len(retained))):
        raise ValueError("Development and test indices must partition the unique source rows.")
    unique = data[retained]
    feature_rows = unique[:, :4]
    if len(np.unique(feature_rows, axis=0)) != len(feature_rows):
        raise ValueError("Duplicate features or conflicting labels require investigation.")
    return unique, development, test


def validation_folds(targets: np.ndarray, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if count < 2 or any(np.sum(targets == label) < count for label in (0, 1)):
        raise ValueError("Each class needs at least one observation per validation fold.")
    generator = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(count)]
    for label in np.unique(targets):
        indices = generator.permutation(np.flatnonzero(targets == label))
        for bucket, chunk in zip(buckets, np.array_split(indices, count)):
            bucket.extend(chunk.tolist())
    return [
        (np.setdiff1d(np.arange(len(targets)), bucket), np.asarray(bucket, dtype=int))
        for bucket in buckets
    ]


def fit_model(
    config: ExperimentConfiguration, features: np.ndarray, labels: np.ndarray, seed: int,
) -> tuple[NeuralNetwork, np.ndarray, float]:
    network = NeuralNetwork(config.architecture, hidden_activations=config.activation, random_seed=seed)
    trainer = PSOANNTrainer(
        network, features, labels, seed=seed,
        **{key: getattr(config, key) for key in (
            "swarm_size", "max_evaluations", "alpha", "beta", "gamma", "delta",
            "epsilon", "informant_count",
        )},
    )
    parameters, fitness = trainer.train()
    return network, parameters, fitness


def predict_saved(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    activations = [str(value) for value in model["activations"]]
    if activations[-1] not in ("logistic", "sigmoid"):
        raise ValueError("Comparison models must have a sigmoid output.")
    network = NeuralNetwork(
        [int(width) for width in model["layer_sizes"]],
        hidden_activations=["sigmoid" if name == "logistic" else name for name in activations[:-1]],
    )
    if len(activations) != len(network.weights):
        raise ValueError("Saved activations do not match the model layers.")
    mean, scale = model["mean"], model["scale"]
    if mean.shape != (4,) or scale.shape != (4,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid saved model scaler.")
    if not np.isfinite(model["parameters"]).all():
        raise ValueError("Saved model parameters must be finite.")
    set_parameter_vector(network, model["parameters"])
    probabilities = network.forward((features - mean) / scale).reshape(-1)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Model produced invalid probabilities.")
    return probabilities


def comparison_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if not np.any(actual == 0) or not np.any(actual == 1):
        raise ValueError("Comparison metrics require both classes.")
    result = binary_metrics(probabilities, actual)
    specificity = float(np.mean(probabilities[actual == 0] < 0.5))
    result["balanced_accuracy"] = (specificity + result["recall"]) / 2
    positives, negatives = probabilities[actual == 1], probabilities[actual == 0]
    result["roc_auc"] = float(np.mean(
        (positives[:, None] > negatives) + 0.5 * (positives[:, None] == negatives)
    ))
    result["brier_score"] = float(np.mean((probabilities - actual) ** 2))
    return result


def comparison_predictions(results: Path, data: np.ndarray, data_digest: str) -> tuple[dict, dict, dict, dict]:
    metadata = json.loads((results / "shared_comparison.json").read_text())
    if data_digest != metadata["data_sha256"]:
        raise ValueError("Shared comparison belongs to a different source dataset.")
    for name, digest in metadata["source_hashes"].items():
        if file_digest(results / name) != digest:
            raise ValueError(f"{name} changed. Regenerate the shared comparison before displaying it.")
    riddhi = load_arrays(results / "best_model.npz")
    maazuddin = load_arrays(results / "maazuddin_shared_model.npz")
    for key in ("retained_original_row_indices", "development_indices", "test_indices"):
        if not np.array_equal(riddhi[key], maazuddin[key]):
            raise ValueError(f"Comparison models do not share {key}.")
    unique, _, test = shared_split(data, riddhi)
    rows = {
        "original_row_index": riddhi["retained_original_row_indices"][test],
        "actual_class": unique[test, -1].astype(int),
        "Riddhi": predict_saved(riddhi, unique[test, :4]),
        "Maazuddin": predict_saved(maazuddin, unique[test, :4]),
    }
    return metadata, riddhi, maazuddin, rows


def main() -> None:
    source = ROOT / "outline" / "data_banknote_authentication.txt"
    experiment_path = RESULTS / "experiment.json"
    experiment = json.loads(experiment_path.read_text())
    if file_digest(source) != experiment["data_sha256"]:
        raise ValueError("Dataset fingerprint differs from Riddhi's experiment.")
    reference = load_arrays(RESULTS / "best_model.npz")
    unique, development, _ = shared_split(load_banknotes(source), reference)
    features, labels = unique[development, :4], unique[development, -1].astype(int)
    folds = validation_folds(labels, experiment["fold_count"], experiment["fold_seed"])
    candidates = []
    for config in CONFIGURATIONS:
        scores = []
        for fold_index, (training, validation) in enumerate(folds, 1):
            train_x, valid_x, _, _ = standardise_features(features[training], features[validation])
            for seed in experiment["search_seeds"]:
                network, _, _ = fit_model(config, train_x, labels[training], seed)
                metrics = binary_metrics(network.forward(valid_x), labels[validation])
                scores.append({"fold": fold_index, "seed": seed, "accuracy": metrics["accuracy"]})
        candidates.append({
            "config": asdict(config), "scores": scores,
            "mean_accuracy": float(np.mean([score["accuracy"] for score in scores])),
            "parameters": total_parameter_count(network),
        })
        print(f"{config.group} / {config.name}: CV accuracy {candidates[-1]['mean_accuracy']:.6%}", flush=True)
    winner = min(candidates, key=lambda row: (-row["mean_accuracy"], row["parameters"]))
    config = ExperimentConfiguration(**winner["config"])
    train_x, _, mean, scale = standardise_features(features, features)
    final_runs = [
        (seed, fit_model(config, train_x, labels, seed))
        for seed in experiment["final_seeds"]
    ]
    seed, (_, parameters, fitness) = max(final_runs, key=lambda run: run[1][2])
    output = RESULTS / "maazuddin_shared_model.npz"
    np.savez_compressed(
        output, parameters=parameters, layer_sizes=np.asarray(config.architecture),
        activations=np.asarray([config.activation] * (len(config.architecture) - 2) + ["sigmoid"]),
        mean=mean, scale=scale,
        **{key: reference[key] for key in (
            "retained_original_row_indices", "development_indices", "test_indices",
        )},
    )
    metadata = {
        "data_sha256": experiment["data_sha256"],
        "source_hashes": {name: file_digest(RESULTS / name) for name in (
            "experiment.json", "best_model.npz", "maazuddin_shared_model.npz",
        )},
        "selection": "Mean development CV accuracy descending, parameter count ascending, configuration order for ties.",
        "fold_seed": experiment["fold_seed"], "fold_count": experiment["fold_count"],
        "search_seeds": experiment["search_seeds"], "candidates": candidates,
        "selected": winner, "selected_final_seed": seed, "training_accuracy": fitness,
        "final_seeds": experiment["final_seeds"],
    }
    (RESULTS / "shared_comparison.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved development-selected model: {config.name}, seed {seed}. No test scores used for selection.")


if __name__ == "__main__":
    main()
