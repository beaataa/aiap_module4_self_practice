"""
trains and compares the classification models for predicting Survive.

two things shape this file. first, the split happens before anything is fitted,
and every step that learns from the data (imputing Creatinine, one hot
encoding, scaling) sits inside a Pipeline, so cross validation refits them on
each fold and nothing from the test set leaks into training.

second, the split is by patient, not by row. the 15000 rows are 297 patients
recorded 32 to 109 times each, so a random row split puts copies of the same
patient on both sides and every model comes out near perfect. run_leakage_check
reproduces that result next to the honest one.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_preparation import PROJECT_ROOT, load_config, prepare_data
from logging_setup import get_logger, setup_logging

log = get_logger("training")


# the majority-class baseline never predicts a survivor, so precision has no
# denominator on every fold it appears in. that is the point of the baseline,
# not a problem to fix
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# the five standard classification metrics, nothing more. balanced accuracy and
# average precision were dropped: precision and recall already say what the
# first one says, and roc_auc already says what the second one says
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def split_features_target(df, config):
    y = df[config["target"]]
    X = df.drop(columns=[config["target"]] + config["drop_for_modelling"])
    groups = df[config["split"]["group_column"]]
    return X, y, groups


def patient_split(X, y, groups, split):
    """holds out one fifth of the patients, keeping the class balance.

    sklearn has no grouped equivalent of train_test_split, so this takes the
    first fold of a StratifiedGroupKFold instead. same effect: whole patients
    on one side or the other, and both sides carry roughly the same share of
    survivors.
    """
    splitter = StratifiedGroupKFold(
        n_splits=split["n_splits"], shuffle=True, random_state=split["random_state"]
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx],
    )


def build_preprocessor(X, preprocessing):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric = Pipeline([
        ("impute", SimpleImputer(
            strategy=preprocessing["numeric_impute_strategy"],
            add_indicator=preprocessing["add_missing_indicator"],
        )),
        # only the logistic model needs this, but scaling the trees too costs
        # nothing and keeps one preprocessor for every model
        ("scale", StandardScaler()),
    ])

    # sparse_output=False because HistGradientBoosting cannot take sparse input.
    # handle_unknown="ignore" so an unseen category at predict time becomes all
    # zeros instead of an error
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def get_models(random_state):
    return {
        # not a real model, just the score to beat. predicts "did not survive"
        # for everyone, which is right 68% of the time and useless
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        # one tree on its own, to show what the ensembles are built out of
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        # many trees on random subsets, voting
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        # trees built one after another, each fixing the last one's errors
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=random_state),
    }


def make_cv(evaluation):
    return StratifiedGroupKFold(
        n_splits=evaluation["cv_folds"],
        shuffle=True,
        random_state=evaluation["random_state"],
    )


def tune_models(models, preprocessor, X_train, y_train, groups_train, tuning, evaluation):
    """grid search the models that have a grid in config, leave the rest alone."""
    cv = make_cv(evaluation)

    tuned, best_params, all_scores = {}, {}, []
    for name, model in models.items():
        grid = tuning["grids"].get(name)
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])

        if not grid:
            tuned[name] = pipe
            continue

        search = GridSearchCV(
            pipe, grid, cv=cv, scoring=evaluation["tuning_metric"], n_jobs=-1
        )
        search.fit(X_train, y_train, groups=groups_train)

        tuned[name] = search.best_estimator_
        best_params[name] = search.best_params_
        log.info("  %-20s %s", name, search.best_params_)

        # keep every combination that was tried, not just the winner
        scores = pd.DataFrame(search.cv_results_)
        scores["model"] = name
        scores["cv_score"] = scores["mean_test_score"]
        scores["cv_score_std"] = scores["std_test_score"]
        scores["params"] = scores["params"].astype(str)
        all_scores.append(
            scores[["model", "params", "cv_score", "cv_score_std", "rank_test_score"]]
        )

    grid_scores = (
        pd.concat(all_scores)
        .sort_values(["model", "rank_test_score"])
        .reset_index(drop=True)
        if all_scores
        else pd.DataFrame()
    )

    return tuned, best_params, grid_scores


def cross_validate_models(pipelines, X_train, y_train, groups_train, evaluation):
    """scores whatever pipelines it is handed, tuned or not.

    this runs after tuning so that the cross validation numbers and the test
    numbers describe the same estimator. scoring the defaults here and the
    tuned version on the test set would make the two columns incomparable.
    """
    cv = make_cv(evaluation)

    rows = []
    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe, X_train, y_train, groups=groups_train, cv=cv, scoring=SCORING
        )

        row = {"model": name}
        row.update({f"cv_{k}": scores[f"test_{k}"].mean() for k in SCORING})
        rows.append(row)

        log.info("  %-20s auc %.3f  f1 %.3f  acc %.3f", name,
                 row["cv_roc_auc"], row["cv_f1"], row["cv_accuracy"])

    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)


def predict_scores(pipe, X):
    """probability of survival, or the decision value if there is no probability."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def evaluate_on_test(fitted, X_test, y_test):
    """the test set is touched once, here, after every choice has been made."""
    rows = []
    for name, pipe in fitted.items():
        pred = pipe.predict(X_test)
        score = predict_scores(pipe, X_test)

        rows.append({
            "model": name,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, score),
        })

    return pd.DataFrame(rows)


def confusion_table(pipe, X_test, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, pipe.predict(X_test)).ravel()

    return pd.DataFrame(
        [
            ("predicted did not survive", tn, fn),
            ("predicted survived", fp, tp),
        ],
        columns=["prediction", "actually did not survive", "actually survived"],
    )


def threshold_profile(pipe, X_test, y_test, thresholds):
    """what the chosen model does as the cut-off moves.

    every headline metric except the two areas under a curve is one row of this
    table. a screening tool that flags patients for review wants recall, and
    recall is bought by lowering the threshold, so the trade is worth printing
    rather than leaving at the 0.5 default.
    """
    score = predict_scores(pipe, X_test)

    rows = []
    for t in thresholds:
        pred = (score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "flagged": int(pred.sum()),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "accuracy": accuracy_score(y_test, pred),
        })

    return pd.DataFrame(rows)


def feature_importances(pipe, name):
    """coefficients for the logistic model, impurity importance for the trees."""
    model = pipe.named_steps["model"]
    features = pipe.named_steps["prep"].get_feature_names_out()

    if hasattr(model, "coef_"):
        values = model.coef_.ravel()
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return None

    return (
        pd.DataFrame({"feature": features, name: values})
        .assign(sort_key=lambda d: d[name].abs())
        .sort_values("sort_key", ascending=False)
        .drop(columns="sort_key")
        .reset_index(drop=True)
    )


def run_stability_check(fitted, X_train, y_train, groups_train, evaluation):
    """repeats the grouped cross validation under different fold assignments.

    297 patients split five ways leaves about 60 per fold, and one fold's AUC
    moves by 0.05 or more just from which patients landed in it. re-drawing the
    folds several times says whether the ranking is a real ordering or the
    accident of one shuffle.
    """
    seeds = range(evaluation["stability_seeds"])

    rows = []
    per_seed = {}
    for name, pipe in fitted.items():
        means = []
        for seed in seeds:
            cv = StratifiedGroupKFold(
                n_splits=evaluation["cv_folds"], shuffle=True, random_state=seed
            )
            scores = cross_val_score(
                clone(pipe), X_train, y_train, groups=groups_train,
                cv=cv, scoring="roc_auc", n_jobs=-1,
            )
            means.append(scores.mean())

        per_seed[name] = means
        rows.append({
            "model": name,
            "mean_auc": np.mean(means),
            "std_auc": np.std(means),
            "worst_auc": np.min(means),
            "best_auc": np.max(means),
        })

    # how often each model came first, across the repeats
    ranking = pd.DataFrame(per_seed)
    wins = ranking.idxmax(axis=1).value_counts()

    out = pd.DataFrame(rows)
    out["times_ranked_first"] = out["model"].map(wins).fillna(0).astype(int)
    out = out.sort_values("mean_auc", ascending=False).reset_index(drop=True)

    for _, r in out.iterrows():
        log.info("  %-20s auc %.3f +- %.3f  first in %d/%d repeats", r["model"],
                 r["mean_auc"], r["std_auc"], r["times_ranked_first"], len(list(seeds)))

    return out


def run_leakage_check(models, preprocessor, X, y, groups, evaluation):
    """the same models scored both ways, to show why the split is by patient.

    ignoring the patient grouping is the single decision that changes the
    answer most here, so the cost of getting it wrong is measured rather than
    asserted.
    """
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", clone(model))])

        by_row = cross_validate(
            pipe, X, y,
            cv=StratifiedKFold(
                n_splits=evaluation["cv_folds"], shuffle=True,
                random_state=evaluation["random_state"],
            ),
            scoring=SCORING,
        )
        by_patient = cross_validate(
            pipe, X, y, groups=groups, cv=make_cv(evaluation), scoring=SCORING
        )

        rows.append({
            "model": name,
            "random_row_split_auc": by_row["test_roc_auc"].mean(),
            "patient_split_auc": by_patient["test_roc_auc"].mean(),
            "random_row_split_accuracy": by_row["test_accuracy"].mean(),
            "patient_split_accuracy": by_patient["test_accuracy"].mean(),
        })
        log.info("  %-20s auc %.3f by row  vs %.3f by patient", name,
                 rows[-1]["random_row_split_auc"], rows[-1]["patient_split_auc"])

    return pd.DataFrame(rows)


def save_model(pipe, name, X, config, comparison):
    """writes the fitted pipeline plus everything predict.py needs to use it.

    the pipeline alone is not enough: whoever loads it has to know which columns
    to pass and in what order, so the schema goes in the same directory.
    """
    model_dir = PROJECT_ROOT / config["evaluation"]["model_dir"]
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "pipeline.joblib"
    joblib.dump(pipe, model_path)

    scores = comparison.loc[comparison["model"] == name].iloc[0]
    metadata = {
        "model": name,
        "target": config["target"],
        "features": list(X.columns),
        "sklearn_estimator": type(pipe.named_steps["model"]).__name__,
        "trained_on_rows": int(len(X)),
        "chosen_by": config["evaluation"]["ranking_metric"],
        "scores": {
            k: round(float(scores[k]), 4)
            for k in comparison.columns if k != "model"
        },
        "default_threshold": 0.5,
        "note": (
            "predict_proba gives the probability of Survive=1. the threshold is a "
            "clinical choice, see results/threshold_profile.csv"
        ),
    }
    schema_path = model_dir / "model_metadata.json"
    schema_path.write_text(json.dumps(metadata, indent=2) + "\n")

    log.info("saved model to %s", model_path)
    log.info("saved schema to %s", schema_path)
    return model_path


def main():
    setup_logging(log_file=PROJECT_ROOT / "logs" / "model_training.log")
    config = load_config()
    log.info("config validated")
    evaluation = config["evaluation"]

    df = prepare_data(config)
    X, y, groups = split_features_target(df, config)
    log.info("%d rows from %d patients", len(X), groups.nunique())

    X_train, X_test, y_train, y_test, groups_train = patient_split(
        X, y, groups, config["split"]
    )
    log.info("train %d rows / %d patients, test %d rows / %d patients",
             len(X_train), groups_train.nunique(), len(X_test),
             groups.nunique() - groups_train.nunique())
    log.info("survived: %.1f%% of training rows, %.1f%% of test rows",
             y_train.mean() * 100, y_test.mean() * 100)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        X, config["preprocessing"]
    )
    log.info("numeric features: %s", numeric_cols)
    log.info("categorical features: %s", categorical_cols)

    models = get_models(evaluation["random_state"])

    best_params = {}
    if config["tuning"]["enabled"]:
        log.info("tuning %d grids", len(config["tuning"]["grids"]))
        fitted, best_params, grid_scores = tune_models(
            models, preprocessor, X_train, y_train, groups_train,
            config["tuning"], evaluation,
        )
    else:
        grid_scores = pd.DataFrame()
        fitted = {
            name: Pipeline([("prep", clone(preprocessor)), ("model", model)])
            for name, model in models.items()
        }

    log.info("cross validating on the training patients, %d folds", evaluation["cv_folds"])
    cv_results = cross_validate_models(
        fitted, X_train, y_train, groups_train, evaluation
    )

    for pipe in fitted.values():
        pipe.fit(X_train, y_train)

    test_results = evaluate_on_test(fitted, X_test, y_test)

    log.info("repeating the cross validation under %d fold draws",
             evaluation["stability_seeds"])
    stability = run_stability_check(fitted, X_train, y_train, groups_train, evaluation)

    comparison = (
        cv_results
        .merge(test_results, on="model")
        .merge(
            stability.rename(columns={
                "mean_auc": "repeated_cv_roc_auc",
                "std_auc": "repeated_cv_roc_auc_std",
            })[["model", "repeated_cv_roc_auc", "repeated_cv_roc_auc_std",
                "times_ranked_first"]],
            on="model",
        )
        .sort_values(evaluation["ranking_metric"], ascending=False)
        .reset_index(drop=True)
    )

    headline = [
        "model", "repeated_cv_roc_auc", "repeated_cv_roc_auc_std",
        "cv_roc_auc", "cv_f1", "cv_accuracy",
        "test_roc_auc", "test_f1", "test_precision", "test_recall",
        "test_accuracy",
    ]
    log.info("model comparison\n%s", comparison[headline].round(3).to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_pipe = fitted[best_name]
    log.info("best by %s: %s", evaluation["ranking_metric"], best_name)

    matrix = confusion_table(best_pipe, X_test, y_test)
    log.info("%s on the test patients at the default 0.5 cut-off\n%s",
             best_name, matrix.to_string(index=False))

    profile = threshold_profile(best_pipe, X_test, y_test, evaluation["thresholds"])
    log.info("what moving the cut-off does\n%s", profile.round(3).to_string(index=False))

    importances = feature_importances(best_pipe, best_name)
    if importances is not None:
        log.info("top features for %s\n%s", best_name,
                 importances.head(15).round(4).to_string(index=False))

    log.info("rescoring on a random row split, for contrast")
    leakage = run_leakage_check(models, preprocessor, X, y, groups, evaluation)

    results_dir = PROJECT_ROOT / evaluation["results_dir"]
    results_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    matrix.to_csv(results_dir / "confusion_matrix.csv", index=False)
    profile.to_csv(results_dir / "threshold_profile.csv", index=False)
    leakage.to_csv(results_dir / "leakage_check.csv", index=False)
    stability.to_csv(results_dir / "stability_check.csv", index=False)
    if importances is not None:
        importances.to_csv(results_dir / "feature_importances.csv", index=False)
    if best_params:
        pd.DataFrame(
            [{"model": k, "best_params": str(v)} for k, v in best_params.items()]
        ).to_csv(results_dir / "best_params.csv", index=False)
    if not grid_scores.empty:
        grid_scores.to_csv(results_dir / "grid_search_results.csv", index=False)
    save_model(best_pipe, best_name, X_train, config, comparison)
    log.info("results saved to %s", results_dir)


if __name__ == "__main__":
    main()
