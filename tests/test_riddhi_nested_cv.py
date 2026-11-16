import csv
import json
from unittest.mock import patch

import numpy as np

from src.evaluation import riddhi_nested_cv as study
from src.evaluation.shared_comparison import load_arrays, load_banknotes, predict_saved, shared_split


def test_notebook_loader_has_no_training_or_artifact_side_effects():
    namespace, digest = study.notebook_implementation()
    assert len(namespace["ARCHITECTURES"]) == 14
    assert len(namespace["REFINEMENT_PSO"]) == 3
    assert namespace["TOP_ARCHITECTURES"] == 3
    assert namespace["SEARCH_SEEDS"] == (11, 29)
    assert "FINAL_PARAMETERS" not in namespace
    assert "test_probabilities" not in namespace
    assert len(digest) == 64
    network = namespace["make_network"](dict(namespace["BASE_PSO"], **namespace["ARCHITECTURES"][0]))
    assert network.layer_sizes == (4, 4, 1)


def test_inner_scaling_uses_inner_training_rows_only():
    namespace, _ = study.notebook_implementation()
    features = np.array([[0., 0], [2, 4], [1000, -1000], [3000, 1000]])
    labels = np.array([0, 1, 0, 1])
    training, validation = np.array([0, 1]), np.array([2, 3])
    train_x, train_y, valid_x, valid_y = study.prepare_folds(
        features, labels, [(training, validation)], namespace["fit_scaler"],
    )[0]
    np.testing.assert_array_equal(train_x, [[-1, -1], [1, 1]])
    np.testing.assert_array_equal(train_y, labels[training])
    np.testing.assert_array_equal(valid_y, labels[validation])
    np.testing.assert_allclose(valid_x, (features[validation] - [1, 2]) / [1, 2])


def test_small_end_to_end_run_never_fits_on_outer_validation_or_holdout(tmp_path):
    namespace, digest = study.notebook_implementation()
    reference = load_arrays(study.RESULTS / "best_model.npz")
    data = load_banknotes(study.ROOT / "outline/data_banknote_authentication.txt")
    unique, development, held_out = shared_split(data, reference)
    features, labels = unique[development, :4], unique[development, -1].astype(int)
    expected_training_sets = []
    for training, validation in namespace["stratified_folds"](labels, count=2, seed=study.OUTER_SEED):
        mean, scale = namespace["fit_scaler"](features[training])
        expected_training_sets.append((features[training] - mean) / scale)
        for inner_training, _ in namespace["stratified_folds"](labels[training], count=3, seed=31415):
            x = features[training][inner_training]
            mean, scale = namespace["fit_scaler"](x)
            expected_training_sets.append((x - mean) / scale)
    original_trainer = namespace["ANNTrainer"]
    calls = []

    class CheckedTrainer(original_trainer):
        def fit(self, features, labels, seed):
            assert any(
                features.shape == expected.shape and np.allclose(features, expected)
                for expected in expected_training_sets
            )
            calls.append(len(labels))
            return super().fit(features, labels, seed)

    namespace["ANNTrainer"] = CheckedTrainer
    namespace["BASE_PSO"] = dict(namespace["BASE_PSO"], particles=8, iterations=2)
    namespace["ARCHITECTURES"] = namespace["ARCHITECTURES"][:2]
    namespace["TOP_ARCHITECTURES"] = 1
    namespace["REFINEMENT_PSO"] = [dict(namespace["BASE_PSO"], iterations=3)]
    with patch.object(study, "OUTER_FOLDS", 2), patch.object(study, "FINAL_SEEDS", (101,)):
        result = study.run_study(namespace, digest, output=tmp_path)
    assert len(calls) == 2 * (3 * 3 * 2 + 1)
    assert len(result["folds"]) == 2
    excluded = set(result["excluded_holdout_original_row_indices"])
    assert excluded == set(reference["retained_original_row_indices"][held_out])
    with (tmp_path / "out_of_fold_predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(development)
    assert not excluded.intersection(int(row["original_row_index"]) for row in rows)


def test_completed_study_oof_predictions_and_selection_reproduce():
    result = json.loads((study.OUTPUT / "results.json").read_text())
    namespace, digest = study.notebook_implementation()
    assert result["implementation_sha256"] == digest
    assert len(result["folds"]) == 5
    reference = load_arrays(study.RESULTS / "best_model.npz")
    data = load_banknotes(study.ROOT / "outline/data_banknote_authentication.txt")
    unique, development, held_out = shared_split(data, reference)
    features, labels = unique[development, :4], unique[development, -1].astype(int)
    with (study.OUTPUT / "out_of_fold_predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1078
    assert len({row["original_row_index"] for row in rows}) == 1078
    probabilities = np.empty(len(rows))
    validation_indices = []
    for fold in result["folds"]:
        training = np.asarray(fold["training_indices"])
        validation = np.asarray(fold["validation_indices"])
        assert not np.intersect1d(training, validation).size
        np.testing.assert_array_equal(np.sort(np.r_[training, validation]), np.arange(len(development)))
        assert not np.intersect1d(development[training], held_out).size
        model = load_arrays(study.OUTPUT / f"outer_fold_{fold['fold']}.npz")
        np.testing.assert_array_equal(model["training_development_indices"], training)
        np.testing.assert_array_equal(model["validation_development_indices"], validation)
        np.testing.assert_allclose(model["mean"], features[training].mean(axis=0))
        np.testing.assert_allclose(model["scale"], features[training].std(axis=0))
        expected_inner = namespace["stratified_folds"](
            labels[training], count=result["inner_folds"], seed=result["inner_seed"],
        )
        for saved, (inner_train, inner_valid) in zip(fold["inner_folds"], expected_inner):
            np.testing.assert_array_equal(saved["training_indices"], inner_train)
            np.testing.assert_array_equal(saved["validation_indices"], inner_valid)
            assert not np.intersect1d(training[inner_train], validation).size
            assert not np.intersect1d(inner_train, inner_valid).size
        candidates = fold["architecture_results"] + fold["refinement_results"]
        assert len(candidates) == 23
        assert all(len(candidate["scores"]) == 6 for candidate in candidates)
        assert min(candidates, key=namespace["ranking_key"]) == fold["selected"]
        restart = min(fold["final_training_costs"], key=lambda run: run["cost"])
        assert restart["seed"] == fold["selected_final_seed"]
        probabilities[validation] = predict_saved(model, features[validation])
        validation_indices.extend(validation)
    np.testing.assert_array_equal(np.sort(validation_indices), np.arange(len(development)))
    np.testing.assert_allclose(
        probabilities, [float(row["probability_class_1"]) for row in rows], atol=1e-14,
    )
    expected = namespace["classification_report"](labels, probabilities)
    assert expected["confusion_matrix"] == result["pooled_out_of_fold"]["confusion_matrix"]
    assert expected["accuracy"] == result["pooled_out_of_fold"]["accuracy"]
    np.testing.assert_array_equal(
        [int(row["original_row_index"]) for row in rows],
        reference["retained_original_row_indices"][development],
    )


def test_notebook_report_executes_against_completed_study():
    notebook = json.loads(study.NOTEBOOK.read_text())
    report_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "nested-cv-report")
    namespace = {
        "DATA_PATH": study.ROOT / "outline/data_banknote_authentication.txt",
        "DATA_SHA256": study.file_digest(study.ROOT / "outline/data_banknote_authentication.txt"),
    }
    with patch("IPython.display.display") as display, patch("plotly.basedatatypes.BaseFigure.show") as show:
        exec(compile("".join(report_cell["source"]), "<nested report>", "exec"), namespace)
    assert show.call_count == 2
    assert display.call_count == 2
    assert len(namespace["nested_fold_frame"]) == 5
    assert len(namespace["nested_predictions"]) == 1078
