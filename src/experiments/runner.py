"""Run configured PSO-ANN experiments without mixing training and test data."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from src.ann.neural_network import NeuralNetwork
from src.data.preprocessing import load_prepared_dataset
from src.evaluation.trainer import PSOANNTrainer
from src.experiments.configurations import ExperimentConfiguration, get_configurations
from src.experiments.metrics import binary_metrics, summarise_runs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "experiments" / "results"
INDIVIDUAL_FIELDS = [
    "experiment_group", "configuration_name", "architecture", "activation", "swarm_size",
    "max_evaluations", "alpha", "beta", "gamma", "delta", "epsilon", "informant_count",
    "run_number", "random_seed", "train_accuracy", "test_accuracy", "test_precision", "test_recall",
    "test_f1_score", "best_training_fitness", "fitness_evaluations", "execution_time_seconds",
]
SUMMARY_FIELDS = INDIVIDUAL_FIELDS[:12] + [
    "mean_train_accuracy", "std_train_accuracy", "mean_test_accuracy", "std_test_accuracy",
    "mean_test_precision", "mean_test_recall", "mean_test_f1_score", "mean_execution_time_seconds",
]


def generate_run_seeds(base_seed: int = 42, runs: int = 10) -> list[int]:
    """Generate deterministic, distinct seeds for independent runs."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer.")
    if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0:
        raise ValueError("runs must be a positive integer.")
    return [base_seed + run_number for run_number in range(runs)]


class ExperimentRunner:
    """Create fresh ANN and PSO instances for each configured independent run."""

    def __init__(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, *, base_seed: int = 42
    ) -> None:
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test, dtype=float)
        self.y_test = np.asarray(y_test)
        if self.X_train.ndim != 2 or self.X_test.ndim != 2 or self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("Training and test features must be two-dimensional with matching feature counts.")
        if self.y_train.ndim != 1 or self.y_test.ndim != 1:
            raise ValueError("Training and test labels must be one-dimensional.")
        if self.y_train.shape[0] != self.X_train.shape[0] or self.y_test.shape[0] != self.X_test.shape[0]:
            raise ValueError("Each label array must match its feature array's sample count.")
        self.base_seed = base_seed

    @classmethod
    def from_processed_data(cls, *, base_seed: int = 42) -> "ExperimentRunner":
        """Load the established processed split for a manual experiment run."""
        data = load_prepared_dataset()
        return cls(data["X_train"], data["y_train"], data["X_test"], data["y_test"], base_seed=base_seed)

    @staticmethod
    def _configuration_fields(configuration: ExperimentConfiguration) -> dict[str, object]:
        return {
            "experiment_group": configuration.group,
            "configuration_name": configuration.name,
            "architecture": str(list(configuration.architecture)),
            "activation": configuration.activation,
            "swarm_size": configuration.swarm_size,
            "max_evaluations": configuration.max_evaluations,
            "alpha": configuration.alpha,
            "beta": configuration.beta,
            "gamma": configuration.gamma,
            "delta": configuration.delta,
            "epsilon": configuration.epsilon,
            "informant_count": configuration.informant_count,
        }

    def run_configuration(self, configuration: ExperimentConfiguration, *, runs: int = 10) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Run one configuration repeatedly, retaining each run and a summary."""
        results: list[dict[str, object]] = []
        for run_number, seed in enumerate(generate_run_seeds(self.base_seed, runs), start=1):
            network = NeuralNetwork(configuration.architecture, hidden_activations=configuration.activation, random_seed=seed)
            trainer = PSOANNTrainer(
                network, self.X_train, self.y_train,
                swarm_size=configuration.swarm_size, max_evaluations=configuration.max_evaluations,
                alpha=configuration.alpha, beta=configuration.beta, gamma=configuration.gamma,
                delta=configuration.delta, epsilon=configuration.epsilon,
                informant_count=configuration.informant_count, seed=seed,
            )
            start_time = perf_counter()
            _, best_fitness = trainer.train()
            execution_time = perf_counter() - start_time
            train_metrics = binary_metrics(network.forward(self.X_train), self.y_train)
            test_metrics = binary_metrics(network.forward(self.X_test), self.y_test)
            result = self._configuration_fields(configuration) | {
                "run_number": run_number,
                "random_seed": seed,
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1_score": test_metrics["f1_score"],
                "best_training_fitness": best_fitness,
                "fitness_evaluations": trainer.optimiser.fitness_evaluations if trainer.optimiser else 0,
                "execution_time_seconds": execution_time,
            }
            results.append(result)
        return results, self._configuration_fields(configuration) | summarise_runs(results)

    def run(
        self, *, groups: list[str] | None = None, runs: int = 10,
        output_directory: Path = DEFAULT_RESULTS_DIRECTORY, overwrite: bool = False,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Run selected configurations and save individual and summary CSV files."""
        individual_results: list[dict[str, object]] = []
        summary_results: list[dict[str, object]] = []
        for configuration in get_configurations(groups):
            results, summary = self.run_configuration(configuration, runs=runs)
            individual_results.extend(results)
            summary_results.append(summary)
        write_results(individual_results, summary_results, output_directory, overwrite=overwrite)
        return individual_results, summary_results


def write_results(
    individual_results: list[dict[str, object]], summary_results: list[dict[str, object]],
    output_directory: Path = DEFAULT_RESULTS_DIRECTORY, *, overwrite: bool = False,
) -> None:
    """Write CSV results, refusing to overwrite either existing file by default."""
    output_directory = Path(output_directory)
    individual_path = output_directory / "individual_results.csv"
    summary_path = output_directory / "summary_results.csv"
    if not overwrite and (individual_path.exists() or summary_path.exists()):
        raise FileExistsError("Result CSV files already exist. Use overwrite=True to replace them.")
    output_directory.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in (
        (individual_path, INDIVIDUAL_FIELDS, individual_results),
        (summary_path, SUMMARY_FIELDS, summary_results),
    ):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
