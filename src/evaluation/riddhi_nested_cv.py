"""Nested CV of the notebook's complete search, without touching its holdout."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.evaluation.shared_comparison import (
    RESULTS, ROOT, file_digest, load_arrays, load_banknotes, shared_split,
)


NOTEBOOK = ROOT / "Riddhi" / "Scripts" / "ann.ipynb"
OUTPUT = RESULTS / "nested_cv"
OUTER_SEED = 271828
OUTER_FOLDS = 5
FINAL_SEEDS = (101, 202, 303)


def notebook_implementation() -> tuple[dict, str]:
    """Load definitions/configuration only, never notebook experiments or writes."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    definitions = {
        "logistic", "FeedForwardANN", "binary_losses", "BinaryCrossEntropyObjective",
        "ParticleSwarmOptimizer", "ANNTrainer", "stratified_folds", "fit_scaler",
        "make_network", "evaluate_config", "ranking_key", "classification_report",
    }
    settings = {"ACTIVATIONS", "SEARCH_SEEDS", "BASE_PSO", "ARCHITECTURES", "TOP_ARCHITECTURES", "REFINEMENT_PSO"}
    nodes = []
    found = set()
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in definitions:
                nodes.append(node)
                found.add(node.name)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in settings:
                    nodes.append(node)
                    found.add(target.id)
    if found != definitions | settings:
        raise ValueError(f"Notebook implementation definitions missing: {(definitions | settings) - found}")
    tree = ast.Module(body=nodes, type_ignores=[])
    namespace = {
        "np": np, "perf_counter": perf_counter, "FOLD_DATA": [],
        "features_development": np.empty((0, 4)),
    }
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    digest = hashlib.sha256(ast.unparse(tree).encode()).hexdigest()
    return namespace, digest


def prepare_folds(features: np.ndarray, labels: np.ndarray, folds: list, fit_scaler) -> list:
    data = []
    for training, validation in folds:
        if np.intersect1d(training, validation).size:
            raise ValueError("Inner training and validation overlap.")
        mean, scale = fit_scaler(features[training])
        data.append((
            (features[training] - mean) / scale, labels[training],
            (features[validation] - mean) / scale, labels[validation],
        ))
    return data


def select_configuration(namespace: dict, fold_data: list, progress=print) -> tuple[dict, list, list]:
    architecture_results = []
    for index, architecture in enumerate(namespace["ARCHITECTURES"], 1):
        config = dict(namespace["BASE_PSO"], **architecture)
        architecture_results.append(namespace["evaluate_config"](config, fold_data=fold_data))
        progress(f"  Architecture {index}/{len(namespace['ARCHITECTURES'])}", flush=True)
    shortlist = sorted(architecture_results, key=namespace["ranking_key"])[:namespace["TOP_ARCHITECTURES"]]
    refinement_results = []
    for index, candidate in enumerate(shortlist, 1):
        for profile_index, profile in enumerate(namespace["REFINEMENT_PSO"], 1):
            previous = candidate["config"]
            config = dict(profile, hidden=previous["hidden"], activations=previous["activations"])
            refinement_results.append(namespace["evaluate_config"](config, fold_data=fold_data))
            progress(f"  Refinement {index}/{len(shortlist)}, profile {profile_index}", flush=True)
    winner = min(architecture_results + refinement_results, key=namespace["ranking_key"])
    return winner, architecture_results, refinement_results


def run_study(namespace: dict, implementation_digest: str, output: Path = OUTPUT) -> dict:
    source = ROOT / "outline/data_banknote_authentication.txt"
    reference = load_arrays(RESULTS / "best_model.npz")
    experiment = json.loads((RESULTS / "experiment.json").read_text())
    if file_digest(source) != experiment["data_sha256"]:
        raise ValueError("Dataset fingerprint differs from the saved experiment.")
    unique, development, held_out = shared_split(load_banknotes(source), reference)
    # Only development features and labels enter the cross-validation procedure.
    features, labels = unique[development, :4], unique[development, -1].astype(int)
    retained = reference["retained_original_row_indices"]
    outer_folds = namespace["stratified_folds"](labels, count=OUTER_FOLDS, seed=OUTER_SEED)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing study: {result_path}")
    results, predictions = [], []
    start = perf_counter()
    for fold_number, (training, validation) in enumerate(outer_folds, 1):
        print(f"Outer fold {fold_number}/{len(outer_folds)}: selecting using inner CV only", flush=True)
        inner_folds = namespace["stratified_folds"](
            labels[training], count=experiment["fold_count"], seed=experiment["fold_seed"],
        )
        fold_data = prepare_folds(
            features[training], labels[training], inner_folds, namespace["fit_scaler"],
        )
        winner, architectures, refinements = select_configuration(namespace, fold_data)
        config = winner["config"]
        network = namespace["make_network"](config)
        mean, scale = namespace["fit_scaler"](features[training])
        train_x = (features[training] - mean) / scale
        trainer = namespace["ANNTrainer"](network, config)
        final_runs = [(seed, trainer.fit(train_x, labels[training], seed)) for seed in FINAL_SEEDS]
        seed, fit = min(final_runs, key=lambda run: run[1]["cost"])
        valid_x = (features[validation] - mean) / scale
        probabilities = network.predict_proba(valid_x, fit["parameters"])
        report = namespace["classification_report"](labels[validation], probabilities)
        report["binary_cross_entropy"] = float(namespace["binary_losses"](
            network, valid_x, labels[validation], fit["parameters"], l2=0,
        )[0])
        report["brier_score"] = float(np.mean((probabilities - labels[validation]) ** 2))
        majority = int(np.argmax(np.bincount(labels[training], minlength=2)))
        report["majority_baseline_accuracy"] = float(np.mean(labels[validation] == majority))
        fold_result = {
            "fold": fold_number, "training_indices": training.tolist(),
            "validation_indices": validation.tolist(),
            "inner_folds": [
                {"training_indices": train.tolist(), "validation_indices": valid.tolist()}
                for train, valid in inner_folds
            ],
            "selected": winner, "architecture_results": architectures,
            "refinement_results": refinements, "selected_final_seed": seed,
            "final_training_costs": [{"seed": s, "cost": run["cost"]} for s, run in final_runs],
            "report": report,
        }
        results.append(fold_result)
        np.savez_compressed(
            output / f"outer_fold_{fold_number}.npz",
            parameters=fit["parameters"], mean=mean, scale=scale,
            layer_sizes=np.asarray(network.layer_sizes), activations=np.asarray(network.activations),
            training_development_indices=training, validation_development_indices=validation,
        )
        for index, probability in zip(validation, probabilities):
            predictions.append({
                "fold": fold_number, "development_index": int(index),
                "original_row_index": int(retained[development[index]]),
                "actual_class": int(labels[index]), "probability_class_1": float(probability),
                "predicted_class": int(probability >= 0.5),
            })
        print(f"Outer fold {fold_number}: {report['accuracy']:.4%}, "
              f"{config['hidden']} {config['activations']}, "
              f"errors {int(np.sum((probabilities >= 0.5) != labels[validation]))}", flush=True)
    predictions.sort(key=lambda row: row["development_index"])
    np.testing.assert_array_equal([row["development_index"] for row in predictions], np.arange(len(development)))
    truth = np.asarray([row["actual_class"] for row in predictions])
    probabilities = np.asarray([row["probability_class_1"] for row in predictions])
    pooled = namespace["classification_report"](truth, probabilities)
    pooled["brier_score"] = float(np.mean((probabilities - truth) ** 2))
    accuracies = np.asarray([fold["report"]["accuracy"] for fold in results])
    result = {
        "owner": "Riddhi", "protocol": "5 outer folds; full architecture and PSO refinement search inside each outer training split.",
        "scope": "Development set only. Existing 270-row holdout excluded. Not fresh external validation.",
        "outer_seed": OUTER_SEED, "outer_folds": OUTER_FOLDS,
        "inner_seed": experiment["fold_seed"], "inner_folds": experiment["fold_count"],
        "search_seeds": list(namespace["SEARCH_SEEDS"]), "final_seeds": list(FINAL_SEEDS),
        "implementation_sha256": implementation_digest, "data_sha256": file_digest(source),
        "development_original_row_indices": retained[development].tolist(),
        "excluded_holdout_original_row_indices": retained[held_out].tolist(),
        "folds": results, "pooled_out_of_fold": pooled,
        "mean_outer_accuracy": float(accuracies.mean()),
        "std_outer_accuracy": float(accuracies.std(ddof=1)),
        "seconds": perf_counter() - start,
        "interpretation": "Fold SD is descriptive, not a confidence interval. Pooled Wilson interval assumes independent observations; folds share training data. Previously used dataset/search space and unknown specimen dependence limit external generalisation claims.",
    }
    with (output / "out_of_fold_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {result_path}; mean outer accuracy {accuracies.mean():.4%}, "
          f"SD {accuracies.std(ddof=1):.4%}; elapsed {result['seconds']:.1f}s", flush=True)
    return result


def main() -> None:
    namespace, digest = notebook_implementation()
    run_study(namespace, digest)


if __name__ == "__main__":
    main()
