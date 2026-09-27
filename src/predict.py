"""
scores new patients with the saved model.

    python src/predict.py data/new_patients.csv
    python src/predict.py data/new_patients.csv --threshold 0.3 --out predictions.csv

the input must be a csv in the same shape as the raw file: the same column
names, the same label styles. it goes through exactly the cleaning and feature
engineering that training used, because that code is imported rather than
copied. the target column is optional, and if it is present the script also
reports how well the model did.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from data_preparation import (
    PROJECT_ROOT,
    add_features,
    add_patient_key,
    drop_unused,
    fix_age,
    load_config,
    normalise_target,
    standardise_categories,
)
from logging_setup import get_logger, setup_logging

log = get_logger("predict")


def load_model(config):
    model_dir = PROJECT_ROOT / config["evaluation"]["model_dir"]
    model_path = model_dir / "pipeline.joblib"
    schema_path = model_dir / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"no model at {model_path}. run python src/model_training.py first"
        )

    pipe = joblib.load(model_path)
    metadata = json.loads(schema_path.read_text()) if schema_path.exists() else {}
    return pipe, metadata


def prepare_new_data(df, config):
    """the training cleaning, minus the steps that need a target.

    prepare_data() cannot be reused directly because it maps the target, and new
    patients do not have one. every other step is the same function object that
    training called, so the two paths cannot drift apart.
    """
    cleaning = config["cleaning"]
    target = config["target"]

    df = drop_unused(df, cleaning["drop_upfront"])
    if target in df.columns:
        df = normalise_target(df, target, cleaning)
    df = fix_age(df, cleaning)
    df = standardise_categories(df, cleaning)
    df = add_features(df, config["features"])
    df = add_patient_key(df, config["patient_key"])

    return df.reset_index(drop=True)


def align_features(df, metadata, config):
    """keeps the columns the model was trained on, in that order."""
    expected = metadata.get("features")
    if not expected:
        return df.drop(
            columns=[config["target"]] + config["drop_for_modelling"], errors="ignore"
        )

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing columns the model needs: {missing}")

    extra = [c for c in df.columns if c not in expected]
    if extra:
        log.info("ignoring columns the model does not use: %s", extra)

    return df[expected]


def score(pipe, X, threshold):
    probability = pipe.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "survival_probability": probability.round(4),
        "predicted_survive": (probability >= threshold).astype(int),
    })


def report_accuracy(predictions, actual):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    pred = predictions["predicted_survive"]
    prob = predictions["survival_probability"]
    log.info("the input had a %s column, so scoring against it:", "Survive")
    log.info("  rows        %d", len(actual))
    log.info("  accuracy    %.3f", accuracy_score(actual, pred))
    log.info("  precision   %.3f", precision_score(actual, pred, zero_division=0))
    log.info("  recall      %.3f", recall_score(actual, pred, zero_division=0))
    log.info("  f1          %.3f", f1_score(actual, pred, zero_division=0))
    if actual.nunique() > 1:
        log.info("  roc auc     %.3f", roc_auc_score(actual, prob))
    else:
        log.info("  roc auc     undefined, the input is all one class")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("input", type=Path, help="csv shaped like the raw data")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="cut-off on the survival probability. defaults to the saved model's 0.5. "
             "see results/threshold_profile.csv for what moving it costs",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="where to write the predictions. defaults to results/predictions.csv",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(log_file=PROJECT_ROOT / "logs" / "predict.log")

    config = load_config()
    pipe, metadata = load_model(config)
    log.info("loaded %s, trained on %s rows",
             metadata.get("model", "model"), metadata.get("trained_on_rows", "?"))

    threshold = args.threshold
    if threshold is None:
        threshold = metadata.get("default_threshold", 0.5)
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    log.info("threshold %.2f", threshold)

    raw = pd.read_csv(args.input)
    log.info("read %d rows from %s", len(raw), args.input)

    df = prepare_new_data(raw, config)
    X = align_features(df, metadata, config)
    predictions = score(pipe, X, threshold)

    if config["id_column"] in raw.columns:
        predictions.insert(0, config["id_column"], raw[config["id_column"]].values)

    flagged = int(predictions["predicted_survive"].sum())
    log.info("predicted survive: %d of %d rows (%.1f%%)",
             flagged, len(predictions), 100 * flagged / len(predictions))

    if config["target"] in df.columns:
        report_accuracy(predictions, df[config["target"]])

    out = args.out or PROJECT_ROOT / config["evaluation"]["results_dir"] / "predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    log.info("wrote %s", out)

    return predictions


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
