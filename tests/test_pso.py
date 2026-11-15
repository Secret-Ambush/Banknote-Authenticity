"""Tests for the Phase 3 Algorithm 39 PSO implementation."""

from __future__ import annotations

import numpy as np
import pytest

from src.pso.algorithm import ParticleSwarmOptimiser
from src.pso.particle import Particle


def sphere_fitness(position: np.ndarray) -> float:
    """A simple maximisation objective with its best value at the origin."""
    return -float(np.sum(position**2))


def test_particle_initialisation() -> None:
    particle = Particle(np.array([1, 2]), np.array([0.1, -0.1]))
    assert np.array_equal(particle.position, [1.0, 2.0])
    assert np.array_equal(particle.velocity, [0.1, -0.1])
    assert np.array_equal(particle.personal_best_position, particle.position)
    assert particle.personal_best_fitness == -np.inf


def test_swarm_initialisation_dimensions() -> None:
    optimiser = ParticleSwarmOptimiser(sphere_fitness, dimensions=4, swarm_size=5, seed=4)
    assert len(optimiser.particles) == 5
    assert all(particle.position.shape == (4,) for particle in optimiser.particles)
    assert all(particle.velocity.shape == (4,) for particle in optimiser.particles)


def test_personal_and_global_best_updates() -> None:
    optimiser = ParticleSwarmOptimiser(sphere_fitness, dimensions=1, swarm_size=1, seed=1)
    particle = optimiser.particles[0]
    particle.position = np.array([2.0])
    optimiser._assess_fitness(particle)
    particle.position = np.array([0.5])
    optimiser._assess_fitness(particle)
    assert particle.personal_best_fitness == -0.25
    assert np.array_equal(particle.personal_best_position, [0.5])
    assert optimiser.global_best_fitness == -0.25
    assert np.array_equal(optimiser.global_best_position, [0.5])


def test_normal_fitness_evaluation_increments_counter() -> None:
    optimiser = ParticleSwarmOptimiser(sphere_fitness, dimensions=1, swarm_size=1, seed=1)
    optimiser._assess_fitness(optimiser.particles[0])
    assert optimiser.fitness_evaluations == 1


@pytest.mark.parametrize("invalid_fitness", [np.nan, np.inf])
def test_non_finite_fitness_counts_before_raising(invalid_fitness: float) -> None:
    optimiser = ParticleSwarmOptimiser(
        lambda position: invalid_fitness, dimensions=1, swarm_size=1, seed=1
    )
    with pytest.raises(ValueError, match="finite"):
        optimiser._assess_fitness(optimiser.particles[0])
    assert optimiser.fitness_evaluations == 1


def test_fitness_exception_counts_before_propagating() -> None:
    def failing_fitness(position: np.ndarray) -> float:
        raise RuntimeError("fitness failed")

    optimiser = ParticleSwarmOptimiser(failing_fitness, dimensions=1, swarm_size=1, seed=1)
    with pytest.raises(RuntimeError, match="fitness failed"):
        optimiser._assess_fitness(optimiser.particles[0])
    assert optimiser.fitness_evaluations == 1


def test_informants_include_the_particle() -> None:
    optimiser = ParticleSwarmOptimiser(
        sphere_fitness, dimensions=2, swarm_size=5, informant_count=3, seed=10
    )
    for index in range(optimiser.swarm_size):
        informants = optimiser.select_informant_indices(index)
        assert index in informants
        assert len(informants) == 3
        assert len(np.unique(informants)) == 3


def test_reproducibility_with_seed() -> None:
    first = ParticleSwarmOptimiser(sphere_fitness, 2, swarm_size=4, max_evaluations=12, seed=42)
    second = ParticleSwarmOptimiser(sphere_fitness, 2, swarm_size=4, max_evaluations=12, seed=42)
    first_best, first_fitness = first.optimise()
    second_best, second_fitness = second.optimise()
    assert np.array_equal(first_best, second_best)
    assert first_fitness == second_fitness


def test_velocity_and_position_update() -> None:
    optimiser = ParticleSwarmOptimiser(
        sphere_fitness,
        dimensions=2,
        swarm_size=1,
        alpha=0.5,
        beta=1.0,
        gamma=1.0,
        delta=1.0,
        epsilon=2.0,
        seed=3,
    )
    particle = optimiser.particles[0]
    particle.position = np.array([1.0, -1.0])
    particle.velocity = np.array([2.0, -4.0])
    optimiser._assess_fitness(particle)
    optimiser._update_velocities()
    optimiser._move_particles()
    assert np.array_equal(particle.velocity, [1.0, -2.0])
    assert np.array_equal(particle.position, [3.0, -5.0])


def test_optimiser_maximises_fitness() -> None:
    optimiser = ParticleSwarmOptimiser(
        lambda position: float(position[0]),
        dimensions=1,
        swarm_size=2,
        max_evaluations=2,
        position_bounds=(-2.0, 2.0),
        seed=8,
    )
    initial_positions = [particle.position[0] for particle in optimiser.particles]
    best_position, best_fitness = optimiser.optimise()
    assert best_fitness == max(initial_positions)
    assert best_position[0] == max(initial_positions)


def test_evaluation_budget_is_not_exceeded() -> None:
    calls = 0

    def counted_fitness(position: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return sphere_fitness(position)

    optimiser = ParticleSwarmOptimiser(counted_fitness, 3, swarm_size=4, max_evaluations=9, seed=2)
    optimiser.optimise()
    assert calls == 8
    assert optimiser.fitness_evaluations == 8
    assert optimiser.fitness_evaluations <= optimiser.max_evaluations


def test_budget_smaller_than_swarm_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least swarm_size"):
        ParticleSwarmOptimiser(sphere_fitness, dimensions=2, swarm_size=4, max_evaluations=3)


def test_budget_equal_to_swarm_completes_one_full_iteration() -> None:
    optimiser = ParticleSwarmOptimiser(sphere_fitness, 2, swarm_size=3, max_evaluations=3, seed=9)
    initial_positions = [particle.position.copy() for particle in optimiser.particles]
    optimiser.optimise()
    assert optimiser.fitness_evaluations == 3
    assert all(particle.personal_best_fitness > -np.inf for particle in optimiser.particles)
    assert any(
        not np.array_equal(particle.position, initial_position)
        for particle, initial_position in zip(optimiser.particles, initial_positions)
    )


def test_incomplete_final_iteration_stops_without_partial_movement() -> None:
    complete_round = ParticleSwarmOptimiser(sphere_fitness, 2, swarm_size=4, max_evaluations=4, seed=5)
    incomplete_final_round = ParticleSwarmOptimiser(
        sphere_fitness, 2, swarm_size=4, max_evaluations=6, seed=5
    )
    complete_round.optimise()
    incomplete_final_round.optimise()
    assert incomplete_final_round.fitness_evaluations == 4
    assert all(
        np.array_equal(left.position, right.position)
        and np.array_equal(left.velocity, right.velocity)
        and left.personal_best_fitness == right.personal_best_fitness
        for left, right in zip(complete_round.particles, incomplete_final_round.particles)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dimensions": 0}, "dimensions"),
        ({"dimensions": 2, "swarm_size": 0}, "swarm_size"),
        ({"dimensions": 2, "max_evaluations": 0}, "max_evaluations"),
        ({"dimensions": 2, "swarm_size": 2, "informant_count": 3}, "informant_count"),
        ({"dimensions": 2, "informant_count": True}, "informant_count"),
        ({"dimensions": 2, "informant_count": False}, "informant_count"),
        ({"dimensions": 2, "alpha": -0.1}, "non-negative"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ParticleSwarmOptimiser(sphere_fitness, **kwargs)
