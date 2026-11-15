"""Tests for the Phase 1 Banknote data pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.inspection import COLUMN_NAMES, FEATURE_COLUMNS, load_raw_dataset, validate_dataset
from src.data.preprocessing import (
    load_prepared_dataset,
    prepare_dataset,
    save_prepared_dataset,
    standardise_features,
    stratified_train_test_split,
)


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """Load the official raw data, skipping only when it has not been downloaded."""
    try:
        return load_raw_dataset()
    except FileNotFoundError:
        pytest.skip("Raw dataset unavailable; run python -m src.data.download first.")


def test_dataset_loading_and_shape(raw_data: pd.DataFrame) -> None:
    assert raw_data.shape == (1372, 5)
    assert list(raw_data.columns) == COLUMN_NAMES
    validate_dataset(raw_data)


def test_features_and_binary_target(raw_data: pd.DataFrame) -> None:
    assert raw_data[FEATURE_COLUMNS].shape == (1372, 4)
    assert set(raw_data["class"].unique()) == {0, 1}
    assert not raw_data.isna().any().any()
    assert np.isfinite(raw_data.to_numpy(dtype=float)).all()


def test_stratified_split_dimensions_and_class_representation(raw_data: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = stratified_train_test_split(raw_data)
    assert X_train.shape == (960, 4)
    assert X_test.shape == (412, 4)
    assert y_train.shape == (960,)
    assert y_test.shape == (412,)
    for label in (0, 1):
        assert (y_train == label).any()
        assert (y_test == label).any()
        assert abs((y_train == label).mean() - (y_test == label).mean()) < 0.01


def test_split_reproducibility_and_seed_effect(raw_data: pd.DataFrame) -> None:
    first = stratified_train_test_split(raw_data, random_seed=42)
    second = stratified_train_test_split(raw_data, random_seed=42)
    changed = stratified_train_test_split(raw_data, random_seed=7)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert not np.array_equal(first[0], changed[0])


def test_standardisation_uses_training_statistics(raw_data: pd.DataFrame) -> None:
    X_train, X_test, _, _ = stratified_train_test_split(raw_data)
    train_scaled, test_scaled, mean, std = standardise_features(X_train, X_test)
    assert np.allclose(train_scaled.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(train_scaled.std(axis=0), 1.0, atol=1e-12)
    assert np.allclose(test_scaled, (X_test - mean) / std)


def test_save_and_load_processed_dataset(raw_data: pd.DataFrame, tmp_path) -> None:
    output_path = save_prepared_dataset(prepare_dataset(raw_data), tmp_path / "banknote_data.npz")
    loaded = load_prepared_dataset(output_path)
    assert {"X_train", "X_test", "y_train", "y_test", "mean", "std"} <= set(loaded)
    assert loaded["X_train"].shape == (960, 4)
    assert np.array_equal(loaded["y_test"], prepare_dataset(raw_data)["y_test"])
