"""Fixed Phase 5 experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass

from src.ann.architecture import ARCHITECTURES, validate_architecture
from src.ann.activations import get_activation


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Describe one reproducible ANN and PSO experiment setting."""

    group: str
    name: str
    architecture: tuple[int, ...]
    activation: str
    swarm_size: int = 20
    max_evaluations: int = 1_000
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    epsilon: float = 1.0
    informant_count: int | None = None

    def __post_init__(self) -> None:
        """Validate values that are independent of the PSO implementation."""
        validate_architecture(self.architecture)
        get_activation(self.activation)
        if self.swarm_size <= 0 or self.max_evaluations < self.swarm_size:
            raise ValueError("max_evaluations must be at least the positive swarm_size.")
        if self.max_evaluations % self.swarm_size != 0:
            raise ValueError("max_evaluations must allow a whole number of swarm evaluation rounds.")


CONFIGURATIONS: tuple[ExperimentConfiguration, ...] = (
    *(ExperimentConfiguration("architecture", name, sizes, "relu") for name, sizes in ARCHITECTURES.items()),
    ExperimentConfiguration("activation", "Sigmoid", (4, 8, 1), "sigmoid"),
    ExperimentConfiguration("activation", "ReLU", (4, 8, 1), "relu"),
    ExperimentConfiguration("activation", "Tanh", (4, 8, 1), "tanh"),
    ExperimentConfiguration("budget", "Swarm 10", (4, 8, 1), "relu", swarm_size=10, max_evaluations=500),
    ExperimentConfiguration("budget", "Swarm 50", (4, 8, 1), "relu", swarm_size=50, max_evaluations=500),
    ExperimentConfiguration("coefficients", "Balanced", (4, 8, 1), "relu", informant_count=3),
    ExperimentConfiguration(
        "coefficients", "Cognitive emphasis", (4, 8, 1), "relu", beta=1.5, gamma=0.75, delta=0.75, informant_count=3
    ),
    ExperimentConfiguration(
        "coefficients", "Social emphasis", (4, 8, 1), "relu", beta=0.75, gamma=1.125, delta=1.125, informant_count=3
    ),
)


def get_configurations(groups: list[str] | None = None) -> tuple[ExperimentConfiguration, ...]:
    """Return all configurations or configurations in the requested experiment groups."""
    available_groups = {configuration.group for configuration in CONFIGURATIONS}
    if groups is None:
        return CONFIGURATIONS
    unknown_groups = set(groups) - available_groups
    if unknown_groups:
        raise ValueError(f"Unknown experiment group(s): {', '.join(sorted(unknown_groups))}.")
    return tuple(configuration for configuration in CONFIGURATIONS if configuration.group in groups)
