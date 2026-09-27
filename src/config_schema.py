"""
validates config.yaml before anything uses it.

without this, a mistyped key surfaces as a KeyError somewhere inside model
training, after the data has already been loaded and cleaned. the point of
checking up front is that the error names the key and says what was expected.
"""

REQUIRED = {
    "data": ["raw", "cleaned"],
    "cleaning": [
        "drop_upfront", "target_map", "age_absolute", "valid_age_range",
        "smoke_case_fix", "ejection_fraction_map", "diabetes_map",
    ],
    "features": [
        "bmi_from", "anaemia_from", "anaemia_below",
        "high_bp_from", "high_bp_at_least", "drop_after",
    ],
    "patient_key": ["from", "column"],
    "split": ["n_splits", "random_state", "group_column"],
    "preprocessing": ["numeric_impute_strategy", "add_missing_indicator"],
    "evaluation": [
        "cv_folds", "random_state", "results_dir", "model_dir", "tuning_metric",
        "ranking_metric", "thresholds", "stability_seeds",
    ],
    "tuning": ["enabled", "grids"],
}

TOP_LEVEL_SCALARS = ["target", "id_column", "drop_for_modelling"]

VALID_IMPUTE = ["mean", "median", "most_frequent", "constant"]


class ConfigError(ValueError):
    """raised with every problem found, not just the first one."""


def _check_structure(config, problems):
    for section, keys in REQUIRED.items():
        if section not in config:
            problems.append(f"missing section: {section}")
            continue
        if not isinstance(config[section], dict):
            problems.append(f"{section} should be a mapping, got {type(config[section]).__name__}")
            continue
        for key in keys:
            if key not in config[section]:
                problems.append(f"missing key: {section}.{key}")

    for key in TOP_LEVEL_SCALARS:
        if key not in config:
            problems.append(f"missing key: {key}")


def _check_values(config, problems):
    preprocessing = config.get("preprocessing", {})
    strategy = preprocessing.get("numeric_impute_strategy")
    if strategy is not None and strategy not in VALID_IMPUTE:
        problems.append(
            f"preprocessing.numeric_impute_strategy is {strategy!r}, "
            f"expected one of {VALID_IMPUTE}"
        )

    indicator = preprocessing.get("add_missing_indicator")
    if indicator is not None and not isinstance(indicator, bool):
        problems.append("preprocessing.add_missing_indicator should be true or false")

    split = config.get("split", {})
    n_splits = split.get("n_splits")
    if isinstance(n_splits, int) and n_splits < 2:
        problems.append(f"split.n_splits is {n_splits}, needs to be at least 2")

    evaluation = config.get("evaluation", {})
    folds = evaluation.get("cv_folds")
    if isinstance(folds, int) and folds < 2:
        problems.append(f"evaluation.cv_folds is {folds}, needs to be at least 2")

    seeds = evaluation.get("stability_seeds")
    if isinstance(seeds, int) and seeds < 1:
        problems.append(f"evaluation.stability_seeds is {seeds}, needs to be at least 1")

    thresholds = evaluation.get("thresholds")
    if isinstance(thresholds, list):
        outside = [t for t in thresholds if not 0 < t < 1]
        if outside:
            problems.append(f"evaluation.thresholds outside (0, 1): {outside}")

    low_high = config.get("cleaning", {}).get("valid_age_range")
    if isinstance(low_high, list) and len(low_high) == 2 and low_high[0] >= low_high[1]:
        problems.append(f"cleaning.valid_age_range is not low-then-high: {low_high}")


def _check_consistency(config, problems):
    # the ranking metric has to be a column the comparison table actually has
    metric = config.get("evaluation", {}).get("ranking_metric")
    known = {
        "repeated_cv_roc_auc", "cv_roc_auc", "cv_f1", "cv_accuracy",
        "cv_precision", "cv_recall", "test_roc_auc", "test_f1",
        "test_accuracy", "test_precision", "test_recall",
    }
    if metric is not None and metric not in known:
        problems.append(
            f"evaluation.ranking_metric is {metric!r}, which is not a column "
            f"the comparison produces. expected one of {sorted(known)}"
        )

    # the group column must be the one patient_key writes
    key_column = config.get("patient_key", {}).get("column")
    group_column = config.get("split", {}).get("group_column")
    if key_column and group_column and key_column != group_column:
        problems.append(
            f"split.group_column is {group_column!r} but patient_key.column is "
            f"{key_column!r}. the split would group on a column that does not exist"
        )

    # and it must be dropped before modelling, or the model gets the answer
    drop = config.get("drop_for_modelling") or []
    if key_column and key_column not in drop:
        problems.append(
            f"patient_key.column {key_column!r} is missing from drop_for_modelling. "
            f"the model would be handed a per-patient identifier"
        )

    if config.get("tuning", {}).get("enabled") and not config["tuning"].get("grids"):
        problems.append("tuning.enabled is true but tuning.grids is empty")


def validate_config(config):
    """returns the config unchanged, or raises ConfigError listing every problem."""
    if not isinstance(config, dict):
        raise ConfigError(f"config should be a mapping, got {type(config).__name__}")

    problems = []
    _check_structure(config, problems)
    _check_values(config, problems)
    _check_consistency(config, problems)

    if problems:
        listed = "\n".join(f"  - {p}" for p in problems)
        raise ConfigError(f"{len(problems)} problem(s) in config.yaml:\n{listed}")

    return config
