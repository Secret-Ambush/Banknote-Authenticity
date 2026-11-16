import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / "ann.ipynb"
RESULTS = NOTEBOOK.parent.parent / "ann_results"


class ResultIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text())
        cls.namespace = {}
        for index in (3, 5, 6, 7):
            source = "".join(notebook["cells"][index]["source"])
            exec(compile(source, f"{NOTEBOOK}:cell {index}", "exec"), cls.namespace)
        diagnostic_cell = next(
            cell for cell in notebook["cells"] if cell.get("id") == "class-balance-diagnostics"
        )
        cls.diagnostic_source = "".join(diagnostic_cell["source"])
        for source in ("".join(notebook["cells"][9]["source"]), cls.diagnostic_source):
            definitions = [
                node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)
                and node.name in ("classification_report", "imbalance_metrics")
            ]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"),
                 cls.namespace)
        cls.experiment = json.loads((RESULTS / "experiment.json").read_text())
        with np.load(RESULTS / "best_model.npz", allow_pickle=False) as archive:
            cls.model = {key: archive[key].copy() for key in archive.files}
        cls.predictions = np.genfromtxt(
            RESULTS / "test_predictions.csv", delimiter=",", names=True,
        )

    def test_splits_features_scaling_and_bundle_agree(self):
        ns, model = self.namespace, self.model
        bundle = json.loads((RESULTS / "riddhi_results.json").read_text())
        self.assertEqual(bundle["experiment"], self.experiment)
        self.assertEqual(ns["DATA_SHA256"], self.experiment["data_sha256"])
        for saved, current in [
            ("retained_original_row_indices", "retained_indices"),
            ("development_indices", "DEVELOPMENT_INDICES"),
            ("test_indices", "TEST_INDICES"),
        ]:
            np.testing.assert_array_equal(model[saved], ns[current])
        train, test = model["development_indices"], model["test_indices"]
        self.assertEqual(np.intersect1d(train, test).size, 0)
        np.testing.assert_allclose(model["mean"], ns["features"][train].mean(axis=0))
        np.testing.assert_allclose(model["scale"], ns["features"][train].std(axis=0))
        train_rows = set(map(tuple, ns["features"][train]))
        self.assertFalse(any(tuple(row) in train_rows for row in ns["features"][test]))
        for index, (training, validation) in enumerate(ns["CV_FOLDS"]):
            self.assertEqual(np.intersect1d(training, validation).size, 0)
            train_x, train_y, valid_x, valid_y = ns["FOLD_DATA"][index]
            mean = ns["features_development"][training].mean(axis=0)
            scale = ns["features_development"][training].std(axis=0)
            np.testing.assert_allclose(
                train_x, (ns["features_development"][training] - mean) / scale,
            )
            np.testing.assert_allclose(
                valid_x, (ns["features_development"][validation] - mean) / scale,
            )
            np.testing.assert_array_equal(train_y, ns["labels_development"][training])
            np.testing.assert_array_equal(valid_y, ns["labels_development"][validation])

    def test_independent_forward_pass_reproduces_saved_predictions_and_metrics(self):
        ns, model = self.namespace, self.model
        test = model["test_indices"]
        outputs = (ns["features"][test] - model["mean"]) / model["scale"]
        parameters = model["parameters"]
        offset = 0
        for fan_in, fan_out, activation in zip(
            model["layer_sizes"][:-1], model["layer_sizes"][1:], model["activations"],
        ):
            size = int(fan_in * fan_out)
            weights = parameters[offset:offset + size].reshape(int(fan_in), int(fan_out))
            offset += size
            biases = parameters[offset:offset + int(fan_out)]
            offset += int(fan_out)
            outputs = outputs @ weights + biases
            if activation == "relu":
                outputs = np.maximum(0, outputs)
            elif activation == "tanh":
                outputs = np.tanh(outputs)
            elif activation in ("logistic", "sigmoid"):
                outputs = np.exp(-np.logaddexp(0, -outputs))
            elif activation != "linear":
                self.fail(f"Unknown saved activation: {activation}")
        self.assertEqual(offset, len(parameters))
        probabilities = outputs[:, 0]
        truth = ns["labels"][test]
        np.testing.assert_array_equal(
            self.predictions["original_row_index"],
            model["retained_original_row_indices"][test],
        )
        np.testing.assert_array_equal(self.predictions["actual_class"], truth)
        np.testing.assert_allclose(
            probabilities, self.predictions["probability_class_1"], atol=1e-14, rtol=1e-10,
        )
        predictions = (probabilities >= float(model["threshold"])).astype(int)
        np.testing.assert_array_equal(predictions, self.predictions["predicted_class"])
        confusion = np.zeros((2, 2), dtype=int)
        np.add.at(confusion, (truth, predictions), 1)
        report = self.experiment["test_report"]
        self.assertEqual(confusion.tolist(), report["confusion_matrix"])
        self.assertEqual(float(np.mean(predictions == truth)), report["accuracy"])
        positive = probabilities[truth == 1, None]
        negative = probabilities[None, truth == 0]
        auc = np.mean((positive > negative) + 0.5 * (positive == negative))
        self.assertEqual(float(auc), report["roc_auc"])

    def test_perfect_and_imperfect_cv_results_are_reproducible(self):
        for hidden_activation in ("relu", "logistic"):
            saved = next(
                result for result in self.experiment["architecture_results"]
                if result["config"]["hidden"] == [4]
                and result["config"]["activations"][0] == hidden_activation
            )
            with self.subTest(activation=hidden_activation):
                recomputed = self.namespace["evaluate_config"](saved["config"])
                self.assertEqual(recomputed["mean_accuracy"], saved["mean_accuracy"])
                np.testing.assert_allclose(recomputed["mean_bce"], saved["mean_bce"], rtol=1e-8)
                self.assertEqual(
                    [score["accuracy"] for score in recomputed["scores"]],
                    [score["accuracy"] for score in saved["scores"]],
                )

    def test_shuffled_training_labels_do_not_produce_perfect_validation_scores(self):
        ns = self.namespace
        config = next(
            result["config"] for result in self.experiment["architecture_results"]
            if result["config"]["hidden"] == [4]
            and result["config"]["activations"][0] == "relu"
        )
        train_x, train_y, valid_x, valid_y = ns["FOLD_DATA"][0]
        shuffled = np.random.default_rng(777).permutation(train_y)
        network = ns["make_network"](config)
        fit = ns["ANNTrainer"](network, config).fit(train_x, shuffled, seed=11)
        probabilities = network.predict_proba(valid_x, fit["parameters"])
        self.assertEqual(probabilities.shape, valid_y.shape)
        accuracy = float(np.mean((probabilities >= 0.5) == valid_y))
        self.assertLess(accuracy, 0.70, f"Shuffled-label control unexpectedly high: {accuracy}")

    def test_imbalance_checks_expose_a_majority_only_classifier(self):
        targets = np.array([0] * 90 + [1] * 10)
        row = self.namespace["class_balance_row"]("Synthetic imbalance", targets)
        self.assertEqual(row["majority_minority_ratio"], 9)
        report = self.namespace["imbalance_metrics"](targets, np.zeros(100))
        self.assertEqual(report["accuracy"], 0.9)
        self.assertEqual(report["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(report["macro_f1"], (180 / 190) / 2)
        self.assertAlmostEqual(report["brier_score"], 0.1)
        self.assertEqual(report["per_class"][1]["recall"], 0)
        self.assertEqual(report["per_class"][1]["precision"], 0)
        perfect = self.namespace["imbalance_metrics"](targets, targets.astype(float))
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            self.assertEqual(perfect[metric], 1)
        self.assertEqual(perfect["brier_score"], 0)
        for invalid in ([], [0, 0], [1, 2], [[0, 1]]):
            with self.subTest(targets=invalid), self.assertRaises(ValueError):
                self.namespace["class_balance_row"]("Invalid split", invalid)

    def test_balance_tables_and_plots_execute_without_retraining(self):
        ns = self.namespace.copy()
        ns["test_labels"] = self.predictions["actual_class"].astype(int)
        ns["test_probabilities"] = self.predictions["probability_class_1"]
        displayed = []
        ns["display"] = displayed.append
        with patch("plotly.basedatatypes.BaseFigure.show") as show:
            exec(compile(self.diagnostic_source, str(NOTEBOOK), "exec"), ns)
        self.assertEqual(show.call_count, 3)
        self.assertEqual(len(displayed), 4)
        self.assertEqual(len(ns["balance_frame"]), 9)
        self.assertEqual(len(ns["per_class_frame"]), 4)
        for fold, row in enumerate(ns["FOLD_BASELINE_ROWS"]):
            _, training_y, _, validation_y = ns["FOLD_DATA"][fold]
            majority = np.argmax(np.bincount(training_y))
            self.assertEqual(row["training_majority_class"], majority)
            self.assertEqual(row["validation_accuracy"], float(np.mean(validation_y == majority)))
        report = ns["diagnostic_results"]["Selected ANN (threshold 0.5)"]
        self.assertEqual(report["balanced_accuracy"], self.experiment["test_report"]["balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
