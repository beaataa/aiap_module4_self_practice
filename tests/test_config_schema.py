"""config.yaml has to be checked before it is used, and the shipped one has to pass."""

import copy

import pytest

from config_schema import ConfigError, validate_config


def test_shipped_config_is_valid(config):
    assert validate_config(config) is config


def test_rejects_non_mapping():
    with pytest.raises(ConfigError, match="should be a mapping"):
        validate_config(["not", "a", "dict"])


def test_reports_missing_section(config):
    broken = copy.deepcopy(config)
    del broken["preprocessing"]
    with pytest.raises(ConfigError, match="missing section: preprocessing"):
        validate_config(broken)


def test_reports_missing_key(config):
    broken = copy.deepcopy(config)
    del broken["features"]["bmi_from"]
    with pytest.raises(ConfigError, match="missing key: features.bmi_from"):
        validate_config(broken)


def test_rejects_unknown_impute_strategy(config):
    broken = copy.deepcopy(config)
    broken["preprocessing"]["numeric_impute_strategy"] = "avarage"
    with pytest.raises(ConfigError, match="numeric_impute_strategy"):
        validate_config(broken)


def test_rejects_ranking_metric_that_is_not_a_column(config):
    """a typo here would otherwise fail deep inside main() after training."""
    broken = copy.deepcopy(config)
    broken["evaluation"]["ranking_metric"] = "test_rmse"
    with pytest.raises(ConfigError, match="ranking_metric"):
        validate_config(broken)


def test_rejects_group_column_that_does_not_match_patient_key(config):
    """this mismatch would silently break the grouped split, the one thing that matters."""
    broken = copy.deepcopy(config)
    broken["split"]["group_column"] = "patient_id"
    with pytest.raises(ConfigError, match="group_column"):
        validate_config(broken)


def test_rejects_patient_key_reaching_the_model(config):
    """if patient_key is not dropped, the model is handed a per-patient identifier."""
    broken = copy.deepcopy(config)
    broken["drop_for_modelling"] = []
    with pytest.raises(ConfigError, match="drop_for_modelling"):
        validate_config(broken)


def test_rejects_too_few_folds(config):
    broken = copy.deepcopy(config)
    broken["evaluation"]["cv_folds"] = 1
    with pytest.raises(ConfigError, match="cv_folds"):
        validate_config(broken)


def test_rejects_thresholds_outside_zero_to_one(config):
    broken = copy.deepcopy(config)
    broken["evaluation"]["thresholds"] = [0.5, 1.5]
    with pytest.raises(ConfigError, match="thresholds"):
        validate_config(broken)


def test_reports_every_problem_at_once(config):
    """one run should list all the problems, not stop at the first."""
    broken = copy.deepcopy(config)
    broken["evaluation"]["cv_folds"] = 0
    broken["preprocessing"]["numeric_impute_strategy"] = "nope"
    broken["drop_for_modelling"] = []
    with pytest.raises(ConfigError) as caught:
        validate_config(broken)
    assert "3 problem(s)" in str(caught.value)
