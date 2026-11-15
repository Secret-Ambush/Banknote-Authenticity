"""Configuration helpers for the coursework ANN architectures."""

from __future__ import annotations

from collections.abc import Sequence


ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "ANN 1": (4, 4, 1),
    "ANN 2": (4, 8, 1),
    "ANN 3": (4, 16, 1),
    "ANN 4": (4, 8, 4, 1),
    "ANN 5": (4, 16, 8, 1),
}


def validate_architecture(layer_sizes: Sequence[int]) -> tuple[int, ...]:
    """Validate and return ANN layer dimensions as an immutable tuple."""
    sizes = tuple(layer_sizes)
    if len(sizes) < 3:
        raise ValueError("An ANN needs an input layer, at least one hidden layer, and an output layer.")
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in sizes):
        raise ValueError("All layer sizes must be positive integers.")
    if sizes[0] != 4:
        raise ValueError("This coursework ANN requires exactly four input features.")
    if sizes[-1] != 1:
        raise ValueError("This coursework ANN requires exactly one output neuron.")
    return sizes


def get_architecture(name: str) -> tuple[int, ...]:
    """Return one of the five named coursework architectures."""
    try:
        return ARCHITECTURES[name.upper()]
    except (AttributeError, KeyError) as error:
        available = ", ".join(ARCHITECTURES)
        raise ValueError(f"Unsupported architecture '{name}'. Choose one of: {available}.") from error
