"""Particle state for the informant-based particle swarm optimiser."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Particle:
    """Store one particle's location, velocity, and best observed location."""

    position: np.ndarray
    velocity: np.ndarray
    personal_best_position: np.ndarray = field(init=False)
    personal_best_fitness: float = field(default=-np.inf, init=False)

    def __post_init__(self) -> None:
        """Store independent floating-point vectors for particle state."""
        self.position = np.asarray(self.position, dtype=float).copy()
        self.velocity = np.asarray(self.velocity, dtype=float).copy()
        if self.position.ndim != 1 or self.velocity.ndim != 1:
            raise ValueError("Particle position and velocity must be one-dimensional vectors.")
        if self.position.shape != self.velocity.shape:
            raise ValueError("Particle position and velocity must have the same shape.")
        self.personal_best_position = self.position.copy()
