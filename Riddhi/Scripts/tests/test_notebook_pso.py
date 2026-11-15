import ast
import json
from pathlib import Path
import unittest

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / "ann.ipynb"
cells = json.loads(NOTEBOOK.read_text())["cells"]
namespace = {"np": np}
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    definitions = [
        node for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"ParticleSwarmOptimizer", "ANNTrainer"}
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
ParticleSwarmOptimizer = namespace["ParticleSwarmOptimizer"]
ANNTrainer = namespace["ANNTrainer"]


def sphere(positions):
    return np.sum(positions ** 2, axis=1)


class NotebookPSOTests(unittest.TestCase):
    def test_disabled_informants_preserve_legacy_random_stream(self):
        initial = np.array([[1., -2.], [-3., 0.5], [0.2, 0.8]])
        fit = ParticleSwarmOptimizer(sphere, iterations=4, seed=12).optimize(initial)
        np.testing.assert_array_equal(
            fit["history"],
            [0.6800000000000002, 0.5575209259689461, 0.48819438588430486,
             0.45478536733244046, 0.08309497351502566],
        )
        np.testing.assert_array_equal(fit["parameters"], [0.28759345326114477, -0.01962088571791265])

    def test_self_only_and_whole_swarm_targets_are_copies(self):
        positions = np.array([[2., 4.], [1., 3.], [-1., -2.]])
        costs = sphere(positions)
        for count, expected in [(1, positions), (3, np.repeat(positions[[2]], 3, axis=0))]:
            optimizer = ParticleSwarmOptimizer(sphere, informant_count=count)
            targets = optimizer._informant_bests(positions, costs, np.random.default_rng(42))
            np.testing.assert_array_equal(targets, expected)
            self.assertFalse(np.shares_memory(targets, positions))

    def test_informants_include_self_and_sample_distinct_others(self):
        class RecordedRandom:
            def __init__(self):
                self.calls = []
                self.generator = np.random.default_rng(7)

            def choice(self, candidates, size, replace):
                self.calls.append((candidates.copy(), size, replace))
                return self.generator.choice(candidates, size=size, replace=replace)

        positions = np.arange(10.).reshape(5, 2)
        costs = np.array([3., 2., 4., 1., 0.])
        random = RecordedRandom()
        optimizer = ParticleSwarmOptimizer(sphere, informant_count=3)
        optimizer._informant_bests(positions, costs, random)
        self.assertEqual(len(random.calls), 5)
        for index, (candidates, size, replace) in enumerate(random.calls):
            self.assertNotIn(index, candidates)
            self.assertEqual(size, 2)
            self.assertFalse(replace)

    def test_one_step_informant_equation(self):
        initial = np.array([[2., 1.], [0.25, 0.5], [-2., 3.]])
        observed = []

        def objective(positions):
            observed.append(positions.copy())
            return sphere(positions)

        rng = np.random.default_rng(19)
        rng.uniform(-0.1, 0.1, initial.shape)
        rng.random(initial.shape)
        rng.random(initial.shape)
        targets = []
        for index in range(len(initial)):
            others = np.delete(np.arange(len(initial)), index)
            selected = np.r_[index, rng.choice(others, size=1, replace=False)]
            targets.append(initial[selected[np.argmin(sphere(initial)[selected])]])
        expected = initial + 0.6 * rng.random(initial.shape) * (np.asarray(targets) - initial)
        ParticleSwarmOptimizer(
            objective, iterations=1, seed=19, c1=0, c2=0, inertia=(0, 0),
            bound=100, velocity_fraction=1, c_informant=0.6, informant_count=2,
        ).optimize(initial)
        np.testing.assert_allclose(observed[1], expected, rtol=0, atol=1e-14)
        self.assertFalse(np.array_equal(observed[1], initial))

    def test_informants_converge_reproducibly_without_input_mutation(self):
        initial = np.random.default_rng(42).uniform(-4, 4, (24, 3))
        original = initial.copy()
        settings = dict(
            seed=12, iterations=120, c_informant=0.5, informant_count=5,
        )
        first = ParticleSwarmOptimizer(sphere, **settings).optimize(initial)
        second = ParticleSwarmOptimizer(sphere, **settings).optimize(initial)
        self.assertLess(first["cost"], 1e-6)
        self.assertTrue(np.all(np.diff(first["history"]) <= 0))
        self.assertEqual(first["evaluations"], 24 * 121)
        self.assertEqual(first["history"].shape, (121,))
        self.assertLessEqual(np.max(np.abs(first["parameters"])), 6)
        np.testing.assert_array_equal(first["history"], second["history"])
        np.testing.assert_array_equal(initial, original)

    def test_invalid_controls_fail_explicitly(self):
        for count in [0, -1, 1.5, True, np.bool_(True)]:
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "informant_count"):
                ParticleSwarmOptimizer(sphere, informant_count=count)
        for coefficient in [-1, np.inf, np.nan]:
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                ParticleSwarmOptimizer(sphere, c_informant=coefficient)
        optimizer = ParticleSwarmOptimizer(sphere, c_informant=0.5, informant_count=4)
        with self.assertRaisesRegex(ValueError, "swarm size"):
            optimizer.optimize(np.ones((3, 2)))
        fit = ParticleSwarmOptimizer(
            sphere, iterations=2, c_informant=0.5, informant_count=1,
        ).optimize(np.ones((1, 2)))
        self.assertEqual(fit["evaluations"], 3)

    def test_trainer_passes_informant_hyperparameters(self):
        class Network:
            def initialize(self, particles, rng):
                return rng.uniform(-1, 1, (particles, 2))

        config = dict(
            particles=3, iterations=2, c1=1., c2=1., inertia=(0.7, 0.4),
            bound=2., velocity_fraction=0.15, l2=0.,
            c_informant=0.5, informant_count=4,
        )
        trainer = ANNTrainer(Network(), config)
        previous = namespace.get("BinaryCrossEntropyObjective")
        namespace["BinaryCrossEntropyObjective"] = lambda *args, **kwargs: sphere
        try:
            with self.assertRaisesRegex(ValueError, "swarm size"):
                trainer.fit(np.zeros((3, 2)), np.zeros(3), seed=12)
            trainer.config["informant_count"] = 2
            fit = trainer.fit(np.zeros((3, 2)), np.zeros(3), seed=12)
            self.assertEqual(fit["evaluations"], 9)
        finally:
            namespace["BinaryCrossEntropyObjective"] = previous


if __name__ == "__main__":
    unittest.main()
