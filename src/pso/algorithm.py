"""Informant-based PSO following Algorithm 39 in *Essentials of Metaheuristics*."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.pso.particle import Particle


class PSO:
    """Maximise a fitness function with Algorithm 39's informant-based PSO."""

    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        dimensions: int,
        *,
        swarm_size: int = 20,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        epsilon: float = 1.0,
        max_evaluations: int = 1_000,
        informant_count: int | None = None,
        position_bounds: tuple[float, float] = (-1.0, 1.0),
        velocity_bounds: tuple[float, float] = (-1.0, 1.0),
        seed: int | None = None,
    ) -> None:
        """Create a PSO instance without evaluating any particle yet."""
        self._validate_inputs(
            fitness_function,
            dimensions,
            swarm_size,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            max_evaluations,
            informant_count,
            position_bounds,
            velocity_bounds,
        )
        self.fitness_function = fitness_function
        self.dimensions = dimensions
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.max_evaluations = max_evaluations
        self.informant_count = min(3, swarm_size) if informant_count is None else informant_count
        self.generator = np.random.default_rng(seed)
        self.particles = [
            Particle(
                self.generator.uniform(*position_bounds, size=dimensions),
                self.generator.uniform(*velocity_bounds, size=dimensions),
            )
            for _ in range(swarm_size)
        ]
        self.global_best_position: np.ndarray | None = None
        self.global_best_fitness = -np.inf
        self.fitness_evaluations = 0

    @staticmethod
    def _validate_inputs(
        fitness_function: Callable[[np.ndarray], float],
        dimensions: int,
        swarm_size: int,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        epsilon: float,
        max_evaluations: int,
        informant_count: int | None,
        position_bounds: tuple[float, float],
        velocity_bounds: tuple[float, float],
    ) -> None:
        """Check configuration values before the swarm is created."""
        if not callable(fitness_function):
            raise ValueError("fitness_function must be callable.")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer.")
        if not isinstance(swarm_size, int) or isinstance(swarm_size, bool) or swarm_size <= 0:
            raise ValueError("swarm_size must be a positive integer.")
        if not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool) or max_evaluations <= 0:
            raise ValueError("max_evaluations must be a positive integer.")
        if max_evaluations < swarm_size:
            raise ValueError("max_evaluations must be at least swarm_size.")
        if informant_count is not None and (
            not isinstance(informant_count, int)
            or isinstance(informant_count, bool)
            or not 1 <= informant_count <= swarm_size
        ):
            raise ValueError("informant_count must be between 1 and swarm_size.")
        if any(not np.isfinite(value) or value < 0 for value in (alpha, beta, gamma, delta, epsilon)):
            raise ValueError("alpha, beta, gamma, delta, and epsilon must be non-negative finite values.")
        for name, bounds in (("position_bounds", position_bounds), ("velocity_bounds", velocity_bounds)):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must contain two finite values in ascending order.")

    def select_informant_indices(self, particle_index: int) -> np.ndarray:
        """Randomly choose informants while always retaining the particle itself."""
        if not 0 <= particle_index < self.swarm_size:
            raise IndexError("particle_index is outside the swarm.")
        other_indices = np.delete(np.arange(self.swarm_size), particle_index)
        chosen_others = self.generator.choice(
            other_indices, size=self.informant_count - 1, replace=False
        )
        return np.concatenate(([particle_index], np.asarray(chosen_others, dtype=int)))

    def _assess_fitness(self, particle: Particle) -> None:
        """Evaluate a particle and update its personal and global records."""
        self.fitness_evaluations += 1
        fitness = float(self.fitness_function(particle.position.copy()))
        if not np.isfinite(fitness):
            raise ValueError("fitness_function must return a finite numeric value.")
        # Algorithm 39, lines 12-15: assess fitness and update best-known locations.
        if fitness > particle.personal_best_fitness:
            particle.personal_best_fitness = fitness
            particle.personal_best_position = particle.position.copy()
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = particle.position.copy()

    def _update_velocities(self) -> None:
        """Apply the Algorithm 39 velocity equation to every particle."""
        assert self.global_best_position is not None
        for index, particle in enumerate(self.particles):
            # Algorithm 39, lines 16-19: find personal, informants', and global bests.
            informant_indices = self.select_informant_indices(index)
            best_informant = max(
                informant_indices,
                key=lambda informant: self.particles[informant].personal_best_fitness,
            )
            informant_best = self.particles[best_informant].personal_best_position
            # Algorithm 39, lines 20-24: sample b, c, d per dimension and update velocity.
            b = self.generator.uniform(0.0, self.beta, size=self.dimensions)
            c = self.generator.uniform(0.0, self.gamma, size=self.dimensions)
            d = self.generator.uniform(0.0, self.delta, size=self.dimensions)
            particle.velocity = (
                self.alpha * particle.velocity
                + b * (particle.personal_best_position - particle.position)
                + c * (informant_best - particle.position)
                + d * (self.global_best_position - particle.position)
            )

    def _move_particles(self) -> None:
        """Move every particle by its scaled velocity."""
        # Algorithm 39, lines 25-26: mutate each particle using its new velocity.
        for particle in self.particles:
            particle.position = particle.position + self.epsilon * particle.velocity

    def optimise(self) -> tuple[np.ndarray, float]:
        """Run PSO until its fitness-evaluation budget is exhausted."""
        # Algorithm 39, lines 7-10: the swarm was randomly created in __init__.
        # Algorithm 39, lines 11 and 27: repeat until the evaluation budget is exhausted.
        while self.fitness_evaluations + self.swarm_size <= self.max_evaluations:
            for particle in self.particles:
                self._assess_fitness(particle)
            self._update_velocities()
            self._move_particles()

        # Algorithm 39, line 28: return the best location and its fitness.
        assert self.global_best_position is not None
        return self.global_best_position.copy(), self.global_best_fitness


# Descriptive alias for callers who prefer the expanded class name.
ParticleSwarmOptimiser = PSO
