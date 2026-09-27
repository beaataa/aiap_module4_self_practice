"""
cleans the raw patient records and builds the engineered features.

this mirrors the data cleaning section of eda.ipynb. everything here is
row-level or uses fixed rules from config.yaml, so it is safe to run on the
whole dataset before splitting. anything that has to learn a value from the
data (imputing Creatinine, encoding, scaling) belongs in the training pipeline
instead, so it can be fitted on the training split only.
"""

from pathlib import Path

import pandas as pd
import yaml

from config_schema import validate_config
from logging_setup import get_logger, setup_logging

log = get_logger("data_prep")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=CONFIG_PATH, validate=True):
    """reads config.yaml and checks it before anything downstream uses it."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return validate_config(config) if validate else config


def load_raw(config):
    return pd.read_csv(PROJECT_ROOT / config["data"]["raw"])


def drop_unused(df, columns):
    # errors="ignore" so a rerun on already cleaned data doesn't blow up
    return df.drop(columns=columns, errors="ignore")


def normalise_target(df, target, cleaning):
    df = df.copy()

    # 0/1 on most rows, No/Yes on 867 of them. read everything as text first so
    # both styles go through the same map
    mapped = df[target].map(lambda v: str(v).strip()).map(cleaning["target_map"])

    unmapped = df.loc[mapped.isna(), target].unique()
    if len(unmapped):
        raise ValueError(f"unexpected values in {target}: {sorted(unmapped)}")

    df[target] = mapped.astype(int)
    return df


def fix_age(df, cleaning):
    df = df.copy()

    # 430 negative ages, sign errors rather than a different kind of patient
    if cleaning["age_absolute"]:
        df["Age"] = df["Age"].abs()

    low, high = cleaning["valid_age_range"]
    outside = df.loc[(df["Age"] < low) | (df["Age"] > high), "Age"].unique()
    if len(outside):
        raise ValueError(f"ages outside {low}-{high} after cleaning: {sorted(outside)}")

    return df


def standardise_categories(df, cleaning):
    df = df.copy()

    # Yes/No in some rows, YES/NO in others
    if cleaning["smoke_case_fix"]:
        df["Smoke"] = df["Smoke"].str.capitalize()

    # L and N are Low and Normal abbreviated
    df["Ejection Fraction"] = df["Ejection Fraction"].replace(
        cleaning["ejection_fraction_map"]
    )

    # Pre-diabetes and Diabetes are one condition written two ways
    df["Diabetes"] = df["Diabetes"].replace(cleaning["diabetes_map"])

    return df


def add_features(df, features):
    df = df.copy()

    height, weight = features["bmi_from"]
    df["bmi"] = df[weight] / (df[height] / 100) ** 2

    # both of these are binary conditions stored as a continuous number. the
    # spread inside each band is noise, the band itself is the signal
    df["anaemia"] = (
        df[features["anaemia_from"]] < features["anaemia_below"]
    ).astype(int)
    df["high_bp"] = (
        df[features["high_bp_from"]] >= features["high_bp_at_least"]
    ).astype(int)

    return drop_unused(df, features["drop_after"])


def add_patient_key(df, patient_key):
    """recovers which patient a row belongs to.

    the same 297 patients are recorded 32 to 109 times each, with the synthetic
    columns redrawn on every copy. the columns below are constant within a
    patient, so pasting them together identifies one. the split needs this, or
    the same patient lands in both train and test.
    """
    df = df.copy()

    key = None
    for column in patient_key["from"]:
        text = df[column].map(lambda v: str(v))
        key = text if key is None else key + "|" + text

    df[patient_key["column"]] = key
    return df


def check_patient_key(df, patient_key, target):
    """the key is only usable if the target is constant within a patient."""
    per_patient = df.groupby(patient_key["column"])[target].nunique()
    split = per_patient[per_patient > 1]
    if len(split):
        raise ValueError(f"{len(split)} patients carry both outcomes")

    return len(per_patient)


def prepare_data(config=None):
    """returns the cleaned dataframe, patient_key included."""
    config = config or load_config()
    cleaning = config["cleaning"]

    df = load_raw(config)
    df = drop_unused(df, cleaning["drop_upfront"])
    df = normalise_target(df, config["target"], cleaning)
    df = fix_age(df, cleaning)
    df = standardise_categories(df, cleaning)
    df = add_features(df, config["features"])
    df = add_patient_key(df, config["patient_key"])

    check_patient_key(df, config["patient_key"], config["target"])

    return df.reset_index(drop=True)


def main():
    setup_logging(log_file=PROJECT_ROOT / "logs" / "data_preparation.log")
    config = load_config()
    log.info("config validated")

    df = prepare_data(config)

    n_patients = df[config["patient_key"]["column"]].nunique()
    out_path = PROJECT_ROOT / config["data"]["cleaned"]
    df.to_csv(out_path, index=False)

    log.info("rows: %d, columns: %d, patients: %d", len(df), len(df.columns), n_patients)
    log.info("survived: %.1f%% of rows", df[config["target"]].mean() * 100)
    log.info("saved to %s", out_path)


if __name__ == "__main__":
    main()
