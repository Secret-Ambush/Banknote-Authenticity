"""Command-line entry point for manually authorised coursework experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.configurations import get_configurations
from src.experiments.runner import DEFAULT_RESULTS_DIRECTORY, ExperimentRunner


def main() -> None:
    """Parse options and run only the requested experiment groups."""
    groups = sorted({configuration.group for configuration in get_configurations()})
    parser = argparse.ArgumentParser(description="Run PSO-ANN coursework experiments.")
    parser.add_argument("--groups", nargs="+", choices=groups, default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_RESULTS_DIRECTORY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    runner = ExperimentRunner.from_processed_data(base_seed=args.base_seed)
    individual, summaries = runner.run(
        groups=args.groups, runs=args.runs, output_directory=args.output_directory, overwrite=args.overwrite
    )
    print(f"Saved {len(individual)} individual results and {len(summaries)} summaries to {args.output_directory}.")


if __name__ == "__main__":
    main()
