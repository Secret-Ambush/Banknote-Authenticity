from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COURSEWORK = ROOT.parent
RESULTS_DIR = ROOT / "ann_results"
MAAZUDDIN_RESULTS_DIR = COURSEWORK / "experiments" / "results"
GROUPS = (
    "activation_results",
    "architecture_results",
    "budget_results",
    "coefficient_results",
    "pilot_results",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Expected at least one result row in {path}.")

    converted: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == "":
                record[key] = None
                continue
            try:
                record[key] = json.loads(value)
            except json.JSONDecodeError:
                record[key] = value
        converted.append(record)
    return converted


def find_riddhi_experiment() -> tuple[Path, dict[str, Any]]:
    candidates = (RESULTS_DIR / "experiment.json",)
    for path in candidates:
        if path.is_file():
            return path, read_json(path)
    raise FileNotFoundError(
        "Could not find ann.ipynb experiment.json. Run the notebook through its final results cell first."
    )


def build_riddhi_results(source: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "owner": "Riddhi",
        "source": "ann.ipynb",
        "source_artifact": str(source.relative_to(ROOT)),
        "protocol": {
            "architecture_metric": "Mean validation accuracy over repeated stratified 3-fold cross-validation",
            "repeats": len(experiment.get("search_seeds", [])),
            "folds": experiment.get("fold_count"),
            "test_set": "One locked holdout, reported only for the selected final architecture",
            "duplicate_policy": experiment.get("duplicate_policy"),
        },
        "experiment": experiment,
    }


def build_maazuddin_results() -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group_name in GROUPS:
        group_dir = MAAZUDDIN_RESULTS_DIR / group_name
        summary_path = group_dir / "summary_results.csv"
        individual_path = group_dir / "individual_results.csv"
        if not summary_path.is_file() or not individual_path.is_file():
            raise FileNotFoundError(
                f"Expected both {summary_path.name} and {individual_path.name} in {group_dir}."
            )
        groups[group_name.removesuffix("_results")] = {
            "summary": read_csv_rows(summary_path),
            "individual": read_csv_rows(individual_path),
        }

    return {
        "owner": "Maazuddin",
        "source": "Biologically-Inspired-Computation-Coursework/experiments/results",
        "protocol": {
            "split": "One fixed stratified 70/30 train/test split",
            "split_seed": 42,
            "runs": 10,
            "test_metrics": "Mean and standard deviation across independent PSO seeds on the fixed test split",
            "training_evaluations": "Configuration-specific; recorded per result row",
        },
        "groups": groups,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    source, experiment = find_riddhi_experiment()
    riddhi_results = build_riddhi_results(source, experiment)
    maazuddin_results = build_maazuddin_results()
    write_json(RESULTS_DIR / "riddhi_results.json", riddhi_results)
    write_json(RESULTS_DIR / "maazuddin_results.json", maazuddin_results)
    print(f"Saved Riddhi's results from {source}.")
    print(f"Saved Maazuddin's results from {MAAZUDDIN_RESULTS_DIR}.")


if __name__ == "__main__":
    main()
