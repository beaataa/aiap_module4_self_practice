"""
the inference path: save a model, load it, score new rows.

the round trip matters more than any single assertion here. training and
prediction have to apply identical cleaning, and the only way to be sure is to
push rows through both and compare.
"""

import json

import pandas as pd
import pytest

from data_preparation import prepare_data
from predict import align_features, load_model, parse_args, prepare_new_data, score


class TestPrepareNewData:
    def test_works_without_a_target_column(self, raw_rows, config):
        """new patients have no outcome yet, which must not break the cleaning."""
        no_target = raw_rows.drop(columns=["Survive"])
        out = prepare_new_data(no_target, config)
        assert "Survive" not in out.columns
        assert "bmi" in out.columns
        assert len(out) == len(raw_rows)

    def test_uses_the_target_when_it_is_there(self, raw_rows, config):
        out = prepare_new_data(raw_rows, config)
        assert set(out["Survive"].unique()) <= {0, 1}

    def test_applies_the_same_cleaning_as_training(self, config):
        """the two paths share the same functions, so they cannot drift apart."""
        raw = pd.read_csv("data/classification_bonus_practice_data.csv").head(500)
        from_training = prepare_data(config).head(0)          # columns only
        from_predict = prepare_new_data(raw, config)
        assert set(from_predict.columns) == set(from_training.columns)

    def test_engineered_features_match_the_training_values(self, config, cleaned):
        raw = pd.read_csv("data/classification_bonus_practice_data.csv").head(50)
        out = prepare_new_data(raw, config)
        pd.testing.assert_series_equal(
            out["bmi"].reset_index(drop=True),
            cleaned["bmi"].head(50).reset_index(drop=True),
        )


class TestAlignFeatures:
    def test_returns_columns_in_the_saved_order(self, raw_rows, config):
        prepared = prepare_new_data(raw_rows, config)
        metadata = {"features": ["bmi", "Age", "Gender"]}
        assert list(align_features(prepared, metadata, config).columns) == \
            ["bmi", "Age", "Gender"]

    def test_raises_when_a_needed_column_is_absent(self, raw_rows, config):
        prepared = prepare_new_data(raw_rows, config)
        metadata = {"features": ["bmi", "a_column_that_does_not_exist"]}
        with pytest.raises(ValueError, match="missing columns"):
            align_features(prepared, metadata, config)

    def test_drops_the_target_and_key_when_no_schema_is_saved(self, raw_rows, config):
        prepared = prepare_new_data(raw_rows, config)
        out = align_features(prepared, {}, config)
        assert config["target"] not in out.columns
        assert config["patient_key"]["column"] not in out.columns


class TestSavedModel:
    """these need models/pipeline.joblib, so they skip if training has not run."""

    @pytest.fixture(autouse=True)
    def _require_model(self, config):
        try:
            self.pipe, self.metadata = load_model(config)
        except FileNotFoundError:
            pytest.skip("no saved model, run python src/model_training.py first")

    def test_metadata_records_what_is_needed_to_reuse_it(self):
        for key in ["model", "target", "features", "default_threshold", "scores"]:
            assert key in self.metadata

    def test_round_trip_scores_real_rows(self, config):
        raw = pd.read_csv("data/classification_bonus_practice_data.csv").head(100)
        X = align_features(prepare_new_data(raw, config), self.metadata, config)
        out = score(self.pipe, X, 0.5)
        assert len(out) == 100
        assert out["survival_probability"].between(0, 1).all()
        assert set(out["predicted_survive"].unique()) <= {0, 1}

    def test_a_lower_threshold_never_flags_fewer(self, config):
        raw = pd.read_csv("data/classification_bonus_practice_data.csv").head(300)
        X = align_features(prepare_new_data(raw, config), self.metadata, config)
        low = score(self.pipe, X, 0.2)["predicted_survive"].sum()
        high = score(self.pipe, X, 0.8)["predicted_survive"].sum()
        assert low >= high

    def test_the_saved_model_is_the_one_the_comparison_chose(self, config):
        comparison = pd.read_csv("results/model_comparison.csv")
        ranked_first = comparison.sort_values(
            config["evaluation"]["ranking_metric"], ascending=False
        ).iloc[0]["model"]
        assert self.metadata["model"] == ranked_first


class TestArgParsing:
    def test_threshold_defaults_to_none_so_the_model_decides(self):
        args = parse_args(["data/x.csv"])
        assert args.threshold is None

    def test_accepts_a_threshold_and_an_output_path(self):
        args = parse_args(["data/x.csv", "--threshold", "0.3", "--out", "p.csv"])
        assert args.threshold == 0.3
        assert str(args.out) == "p.csv"
