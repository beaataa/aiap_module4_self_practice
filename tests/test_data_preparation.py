"""the cleaning rules, each tested against the case that motivated it."""

import numpy as np
import pandas as pd
import pytest

from data_preparation import (
    add_features,
    add_patient_key,
    check_patient_key,
    drop_unused,
    fix_age,
    normalise_target,
    prepare_data,
    standardise_categories,
)


class TestNormaliseTarget:
    def test_folds_both_encodings_into_0_and_1(self, raw_rows, config):
        out = normalise_target(raw_rows, "Survive", config["cleaning"])
        assert list(out["Survive"]) == [0, 1, 0, 1]
        assert out["Survive"].dtype == int

    def test_raises_on_a_value_it_does_not_recognise(self, raw_rows, config):
        raw_rows.loc[0, "Survive"] = "Maybe"
        with pytest.raises(ValueError, match="unexpected values in Survive"):
            normalise_target(raw_rows, "Survive", config["cleaning"])


class TestFixAge:
    def test_flips_negative_ages(self, raw_rows, config):
        out = fix_age(raw_rows, config["cleaning"])
        assert out.loc[1, "Age"] == 55

    def test_leaves_positive_ages_alone(self, raw_rows, config):
        out = fix_age(raw_rows, config["cleaning"])
        assert list(out["Age"]) == [60, 55, 70, 45]

    def test_raises_when_an_age_is_still_impossible(self, raw_rows, config):
        """a value abs() cannot rescue should stop the run, not pass silently."""
        raw_rows.loc[0, "Age"] = 7
        with pytest.raises(ValueError, match="ages outside 40-95"):
            fix_age(raw_rows, config["cleaning"])


class TestStandardiseCategories:
    def test_folds_smoke_case(self, raw_rows, config):
        out = standardise_categories(raw_rows, config["cleaning"])
        assert set(out["Smoke"]) == {"Yes", "No"}

    def test_expands_abbreviated_ejection_fraction(self, raw_rows, config):
        out = standardise_categories(raw_rows, config["cleaning"])
        assert set(out["Ejection Fraction"]) == {"Low", "Normal", "High"}

    def test_merges_prediabetes_into_diabetes(self, raw_rows, config):
        out = standardise_categories(raw_rows, config["cleaning"])
        assert set(out["Diabetes"]) == {"Normal", "Diabetes"}
        assert "Pre-diabetes" not in set(out["Diabetes"])


class TestAddFeatures:
    def test_bmi_is_weight_over_height_squared(self, raw_rows, config):
        out = add_features(raw_rows, config["features"])
        assert out.loc[0, "bmi"] == pytest.approx(70 / 1.70 ** 2)

    def test_anaemia_flag_splits_at_12(self, raw_rows, config):
        """11.9 is anaemic, 12.0 is not. the boundary is the whole feature."""
        out = add_features(raw_rows, config["features"])
        assert out.loc[2, "anaemia"] == 1   # 11.9
        assert out.loc[3, "anaemia"] == 0   # 12.0
        assert out.loc[0, "anaemia"] == 0   # 14.0

    def test_high_bp_flag_splits_at_120(self, raw_rows, config):
        out = add_features(raw_rows, config["features"])
        assert out.loc[3, "high_bp"] == 0   # 119
        assert out.loc[2, "high_bp"] == 1   # 120
        assert out.loc[1, "high_bp"] == 1   # 150

    def test_drops_the_columns_it_replaces(self, raw_rows, config):
        out = add_features(raw_rows, config["features"])
        for column in ["Height", "Weight", "Hemoglobin", "Blood Pressure"]:
            assert column not in out.columns


class TestPatientKey:
    def test_key_is_built_from_the_configured_columns(self, raw_rows, config):
        out = add_patient_key(raw_rows, config["patient_key"])
        assert out.loc[0, "patient_key"] == "60|140|250000.0|100"

    def test_identical_patients_share_a_key(self, raw_rows, config):
        doubled = pd.concat([raw_rows, raw_rows.iloc[[0]]], ignore_index=True)
        out = add_patient_key(doubled, config["patient_key"])
        assert out.loc[0, "patient_key"] == out.loc[4, "patient_key"]

    def test_check_passes_when_the_outcome_is_constant(self, config):
        df = pd.DataFrame({"patient_key": ["a", "a", "b"], "Survive": [1, 1, 0]})
        assert check_patient_key(df, config["patient_key"], "Survive") == 2

    def test_check_raises_when_one_patient_carries_both_outcomes(self, config):
        """if this ever fires, the fingerprint is not identifying patients."""
        df = pd.DataFrame({"patient_key": ["a", "a"], "Survive": [0, 1]})
        with pytest.raises(ValueError, match="carry both outcomes"):
            check_patient_key(df, config["patient_key"], "Survive")


class TestDropUnused:
    def test_is_safe_to_run_twice(self):
        """rerunning on already cleaned data must not raise."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        once = drop_unused(df, ["a"])
        assert list(drop_unused(once, ["a"]).columns) == ["b"]


class TestPreparedDataset:
    """properties of the real output, the ones the rest of the pipeline assumes."""

    def test_row_count_is_unchanged(self, cleaned):
        assert len(cleaned) == 15000

    def test_target_is_binary_integers(self, cleaned):
        assert set(cleaned["Survive"].unique()) == {0, 1}

    def test_the_decoy_columns_are_gone(self, cleaned):
        assert "Favorite color" not in cleaned.columns
        assert "ID" not in cleaned.columns

    def test_no_negative_ages_survive(self, cleaned):
        assert cleaned["Age"].min() >= 40

    def test_creatinine_is_still_missing_where_it_was(self, cleaned):
        """imputing during cleaning would use the test rows to compute the median."""
        assert cleaned["Creatinine"].isna().sum() == 499

    def test_there_are_297_patients(self, cleaned):
        assert cleaned["patient_key"].nunique() == 297

    def test_every_patient_has_one_outcome(self, cleaned):
        assert (cleaned.groupby("patient_key")["Survive"].nunique() == 1).all()

    def test_bmi_is_constant_within_a_patient(self, cleaned):
        """the finding that justifies building bmi and dropping height and weight."""
        spread = cleaned.groupby("patient_key")["bmi"].std().mean()
        assert spread < 0.5

    def test_flags_are_zero_or_one(self, cleaned):
        for column in ["anaemia", "high_bp"]:
            assert set(cleaned[column].unique()) <= {0, 1}
