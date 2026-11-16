import json
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dashboard
from src.evaluation.shared_comparison import (
    RESULTS, ROOT, comparison_metrics, comparison_predictions, file_digest,
    fit_model, load_arrays, load_banknotes, predict_saved, shared_split, validation_folds,
)
from src.data.preprocessing import standardise_features
from src.experiments.configurations import ExperimentConfiguration
from src.experiments.metrics import binary_metrics


def load_comparison():
    source = ROOT / "outline" / "data_banknote_authentication.txt"
    return comparison_predictions(RESULTS, load_banknotes(source), file_digest(source))


def test_shared_rows_predictions_and_scaling():
    metadata, riddhi, maazuddin, rows = load_comparison()
    data = load_banknotes(ROOT / "outline" / "data_banknote_authentication.txt")
    unique, development, test = shared_split(data, riddhi)
    assert len(test) == 270
    assert not np.intersect1d(development, test).size
    for model in (riddhi, maazuddin):
        np.testing.assert_array_equal(model["development_indices"], development)
        np.testing.assert_array_equal(model["test_indices"], test)
        np.testing.assert_allclose(model["mean"], unique[development, :4].mean(axis=0))
        np.testing.assert_allclose(model["scale"], unique[development, :4].std(axis=0))
    saved_predictions = pd.read_csv(RESULTS / "test_predictions.csv").set_index("original_row_index")
    np.testing.assert_array_equal(rows["actual_class"], unique[test, -1].astype(int))
    np.testing.assert_allclose(
        rows["Riddhi"], saved_predictions.loc[rows["original_row_index"], "probability_class_1"],
        atol=1e-14,
    )
    expected = min(metadata["candidates"], key=lambda row: (-row["mean_accuracy"], row["parameters"]))
    assert metadata["selected"] == expected


def test_matching_notebook_fold_definition_and_no_overlap():
    notebook = json.loads((ROOT / "Riddhi/Scripts/ann.ipynb").read_text())
    # Read only the split function; do not execute notebook training or artifact writes.
    import ast
    nodes = ast.parse("".join(notebook["cells"][6]["source"])).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == "stratified_folds")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook folds>", "exec"), namespace)
    data = load_banknotes(ROOT / "outline/data_banknote_authentication.txt")
    unique, development, _ = shared_split(data, load_arrays(RESULTS / "best_model.npz"))
    labels = unique[development, -1].astype(int)
    metadata = json.loads((RESULTS / "shared_comparison.json").read_text())
    actual = validation_folds(labels, metadata["fold_count"], metadata["fold_seed"])
    expected = namespace["stratified_folds"](labels, metadata["fold_count"], metadata["fold_seed"])
    for (training, validation), (expected_train, expected_valid) in zip(actual, expected):
        np.testing.assert_array_equal(training, expected_train)
        np.testing.assert_array_equal(validation, expected_valid)
        assert not np.intersect1d(training, validation).size
    np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in actual])), np.arange(len(labels)))

def test_selected_cv_scores_and_final_weights_are_reproducible():
    metadata, reference, saved, _ = load_comparison()
    data = load_banknotes(ROOT / "outline/data_banknote_authentication.txt")
    unique, development, _ = shared_split(data, reference)
    features, labels = unique[development, :4], unique[development, -1].astype(int)
    config = ExperimentConfiguration(**metadata["selected"]["config"])
    scores = []
    for training, validation in validation_folds(labels, metadata["fold_count"], metadata["fold_seed"]):
        train_x, valid_x, _, _ = standardise_features(features[training], features[validation])
        for seed in metadata["search_seeds"]:
            network, _, _ = fit_model(config, train_x, labels[training], seed)
            scores.append(binary_metrics(network.forward(valid_x), labels[validation])["accuracy"])
    np.testing.assert_array_equal(scores, [score["accuracy"] for score in metadata["selected"]["scores"]])
    assert float(np.mean(scores)) == metadata["selected"]["mean_accuracy"]
    train_x = (features - saved["mean"]) / saved["scale"]
    _, parameters, fitness = fit_model(config, train_x, labels, metadata["selected_final_seed"])
    np.testing.assert_array_equal(parameters, saved["parameters"])
    assert fitness == metadata["training_accuracy"]


def test_model_selection_never_trains_or_scores_on_test(tmp_path):
    import src.evaluation.shared_comparison as module
    source = ROOT / "outline/data_banknote_authentication.txt"
    reference = load_arrays(RESULTS / "best_model.npz")
    unique, development, test = shared_split(load_banknotes(source), reference)
    experiment = json.loads((RESULTS / "experiment.json").read_text())
    expected_training_sets = []
    for train, _ in validation_folds(
        unique[development, -1].astype(int), experiment["fold_count"], experiment["fold_seed"],
    ):
        train_x = unique[development[train], :4]
        expected_training_sets.append((train_x - train_x.mean(axis=0)) / train_x.std(axis=0))
    expected_training_sets.append(
        (unique[development, :4] - reference["mean"]) / reference["scale"],
    )
    original_fit = module.fit_model
    original_metrics = module.binary_metrics

    def checked_fit(config, features, labels, seed):
        assert any(
            candidate.shape == features.shape and np.allclose(candidate, features)
            for candidate in expected_training_sets
        )
        return original_fit(config, features, labels, seed)

    def checked_metrics(probabilities, labels):
        assert len(labels) in [len(validation) for _, validation in validation_folds(
            unique[development, -1].astype(int), experiment["fold_count"], experiment["fold_seed"],
        )]
        assert len(labels) != len(test)
        return original_metrics(probabilities, labels)

    for name in ("best_model.npz", "experiment.json"):
        shutil.copyfile(RESULTS / name, tmp_path / name)
    with patch.object(module, "RESULTS", tmp_path), patch.object(module, "CONFIGURATIONS", module.CONFIGURATIONS[:1]), patch.object(module, "fit_model", side_effect=checked_fit), patch.object(module, "binary_metrics", side_effect=checked_metrics):
        module.main()
    trained = load_arrays(tmp_path / "maazuddin_shared_model.npz")
    np.testing.assert_array_equal(trained["test_indices"], test)


def test_stale_model_is_rejected(tmp_path):
    for name in ("best_model.npz", "maazuddin_shared_model.npz", "experiment.json", "shared_comparison.json"):
        shutil.copyfile(RESULTS / name, tmp_path / name)
    with (tmp_path / "best_model.npz").open("ab") as stream:
        stream.write(b"changed")
    source = ROOT / "outline/data_banknote_authentication.txt"
    with pytest.raises(ValueError, match="Regenerate"):
        comparison_predictions(tmp_path, load_banknotes(source), file_digest(source))


def test_invalid_split_and_scaler_are_rejected():
    model = load_arrays(RESULTS / "best_model.npz")
    invalid = {**model, "test_indices": model["development_indices"][:270]}
    data = load_banknotes(ROOT / "outline/data_banknote_authentication.txt")
    with pytest.raises(ValueError, match="partition"):
        shared_split(data, invalid)
    with pytest.raises(ValueError, match="scaler"):
        predict_saved({**model, "scale": np.zeros(4)}, data[:1, :4])


def test_tied_roc_scores_and_imbalance_metrics():
    actual = np.array([0, 0, 1, 1])
    probabilities = np.full(4, 0.5)
    scores = comparison_metrics(actual, probabilities)
    assert scores["roc_auc"] == 0.5
    assert scores["balanced_accuracy"] == 0.5
    assert scores["brier_score"] == 0.25
    fpr, tpr = dashboard.roc_points(actual, probabilities)
    np.testing.assert_array_equal(fpr, [0, 1])
    np.testing.assert_array_equal(tpr, [0, 1])


def test_architecture_diagrams_include_every_neuron_and_edge():
    for name in ("best_model.npz", "maazuddin_shared_model.npz"):
        model = load_arrays(RESULTS / name)
        figure = dashboard.architecture_figure(model, "Test")
        layers = model["layer_sizes"]
        assert sum(len(trace.x) for trace in figure.data[1:]) == sum(layers)
        assert len(figure.data[0].x) == 3 * sum(a * b for a, b in zip(layers[:-1], layers[1:]))
        assert len(figure.layout.annotations) == len(layers)


def test_dashboard_summarises_all_pso_configurations_without_dropdown():
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=45).run()
    assert not app.exception
    assert not app.error
    assert [tab.label for tab in app.tabs] == [
        "01  Data explorer", "02  Shared-test model comparison", "03  PSO deep dive",
    ]
    assert not any(widget.key == "pso_configuration" for widget in app.selectbox)
    text = "\n".join(item.value for item in app.markdown)
    assert "β = **0.75 to 1.5**" in text
    assert "γ = **0.75 to 1.125**" in text
    assert "δ = **0.75 to 1.125**" in text
    assert "not a decrease over iterations" in text
    configs = next(frame.value for frame in app.dataframe if "Evaluation budget" in frame.value.columns)
    assert len(configs) == 18
    assert set(configs["Inertia weight"]) == {0.5}
    assert set(configs["Informants"]) == {3}


def test_dashboard_consolidates_tabs_and_pairs_sample_rows():
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=45).run()
    assert not app.exception
    assert not app.error
    selector = next(widget for widget in app.selectbox if widget.label.startswith("Inspect a shared test sample"))
    assert len(selector.options) == 270
    selector.select_index(269).run()
    assert not app.exception
    assert not app.error
    expected_id = int(selector.options[269])
    selected_tables = [
        frame.value for frame in app.dataframe
        if "Source row (0-based)" in frame.value.columns and len(frame.value) == 1
    ]
    assert len(selected_tables) == 1
    assert int(selected_tables[0]["Source row (0-based)"].iloc[0]) == expected_id
