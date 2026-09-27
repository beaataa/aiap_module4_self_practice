"""
the split and the preprocessing, which is where leakage would come from.

the first class is the important one. if patients can appear on both sides of
the split, every score in the project is meaningless, so that invariant gets
tested directly rather than assumed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from model_training import (
    build_preprocessor,
    confusion_table,
    feature_importances,
    get_models,
    patient_split,
    split_features_target,
    threshold_profile,
)


@pytest.fixture(scope="module")
def split(cleaned, config):
    X, y, groups = split_features_target(cleaned, config)
    return (X, y, groups) + patient_split(X, y, groups, config["split"])


class TestPatientSplit:
    def test_no_patient_appears_on_both_sides(self, split):
        """the invariant the whole project rests on."""
        X, y, groups, X_train, X_test, y_train, y_test, groups_train = split
        test_patients = set(groups.loc[X_test.index])
        train_patients = set(groups_train)
        assert test_patients & train_patients == set()

    def test_every_row_lands_somewhere_exactly_once(self, split):
        X, y, groups, X_train, X_test, *_ = split
        assert len(X_train) + len(X_test) == len(X)
        assert set(X_train.index) & set(X_test.index) == set()

    def test_test_share_is_about_one_fifth(self, split):
        X, y, groups, X_train, X_test, *_ = split
        assert 0.15 < len(X_test) / len(X) < 0.25

    def test_class_balance_is_preserved_on_both_sides(self, split):
        """stratified as well as grouped. the two are not alternatives."""
        X, y, groups, X_train, X_test, y_train, y_test, _ = split
        assert abs(y_train.mean() - y.mean()) < 0.05
        assert abs(y_test.mean() - y.mean()) < 0.05

    def test_both_sides_contain_both_classes(self, split):
        *_, y_train, y_test, _ = split
        assert y_train.nunique() == 2
        assert y_test.nunique() == 2

    def test_is_reproducible(self, cleaned, config):
        X, y, groups = split_features_target(cleaned, config)
        first = patient_split(X, y, groups, config["split"])[1].index
        second = patient_split(X, y, groups, config["split"])[1].index
        assert list(first) == list(second)


class TestSplitFeaturesTarget:
    def test_the_target_is_not_among_the_features(self, cleaned, config):
        X, y, groups = split_features_target(cleaned, config)
        assert config["target"] not in X.columns

    def test_the_patient_key_is_not_among_the_features(self, cleaned, config):
        """handing the model a per-patient id would be handing it the answer."""
        X, y, groups = split_features_target(cleaned, config)
        assert config["patient_key"]["column"] not in X.columns

    def test_groups_line_up_with_the_rows(self, cleaned, config):
        X, y, groups = split_features_target(cleaned, config)
        assert len(groups) == len(X)
        assert list(groups.index) == list(X.index)


class TestPreprocessor:
    def test_splits_numeric_from_categorical(self, cleaned, config):
        X, _, _ = split_features_target(cleaned, config)
        _, numeric, categorical = build_preprocessor(X, config["preprocessing"])
        assert "bmi" in numeric
        assert "Gender" in categorical
        assert set(numeric) & set(categorical) == set()
        assert len(numeric) + len(categorical) == len(X.columns)

    def test_fills_missing_values(self, cleaned, config):
        X, y, _ = split_features_target(cleaned, config)
        prep, _, _ = build_preprocessor(X, config["preprocessing"])
        out = prep.fit_transform(X)
        assert not np.isnan(out).any()

    def test_does_not_add_a_missing_indicator(self, cleaned, config):
        """all 499 gaps sit in non-survivors, so an indicator would be the target."""
        assert config["preprocessing"]["add_missing_indicator"] is False
        X, _, _ = split_features_target(cleaned, config)
        prep, _, _ = build_preprocessor(X, config["preprocessing"])
        prep.fit(X)
        assert not any("missingindicator" in f.lower() for f in prep.get_feature_names_out())

    def test_survives_a_category_it_has_never_seen(self, cleaned, config):
        """handle_unknown="ignore", so a new label becomes zeros instead of a crash."""
        X, _, _ = split_features_target(cleaned, config)
        prep, _, _ = build_preprocessor(X, config["preprocessing"])
        prep.fit(X.iloc[:1000])
        unseen = X.iloc[[0]].copy()
        unseen["Gender"] = "Unknown"
        assert prep.transform(unseen).shape[1] == prep.transform(X.iloc[[0]]).shape[1]


class TestModels:
    def test_includes_a_baseline_and_at_least_three_real_models(self, config):
        models = get_models(config["evaluation"]["random_state"])
        assert any("Baseline" in name for name in models)
        assert len(models) - 1 >= 3

    def test_every_model_can_give_a_probability(self, config):
        """threshold tuning and roc auc both need predict_proba."""
        for name, model in get_models(config["evaluation"]["random_state"]).items():
            assert hasattr(model, "predict_proba"), name


@pytest.fixture(scope="module")
def fitted(cleaned, config):
    X, y, groups = split_features_target(cleaned, config)
    prep, _, _ = build_preprocessor(X, config["preprocessing"])
    pipe = Pipeline([
        ("prep", prep),
        ("model", DummyClassifier(strategy="stratified", random_state=0)),
    ])
    return pipe.fit(X, y), X, y


class TestThresholdProfile:
    def test_raising_the_threshold_never_flags_more_patients(self, fitted, config):
        pipe, X, y = fitted
        profile = threshold_profile(pipe, X, y, config["evaluation"]["thresholds"])
        assert profile["flagged"].is_monotonic_decreasing

    def test_one_row_per_threshold(self, fitted, config):
        pipe, X, y = fitted
        thresholds = config["evaluation"]["thresholds"]
        profile = threshold_profile(pipe, X, y, thresholds)
        assert list(profile["threshold"]) == list(thresholds)

    def test_confusion_table_totals_match_the_row_count(self, fitted):
        pipe, X, y = fitted
        table = confusion_table(pipe, X, y)
        counted = table[["actually did not survive", "actually survived"]].values.sum()
        assert counted == len(y)


class TestFeatureImportances:
    def test_returns_none_when_the_model_has_none(self, cleaned, config):
        X, y, _ = split_features_target(cleaned, config)
        prep, _, _ = build_preprocessor(X, config["preprocessing"])
        pipe = Pipeline([("prep", prep), ("model", DummyClassifier())]).fit(X, y)
        assert feature_importances(pipe, "Dummy") is None
