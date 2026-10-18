"""Shared models, date groups, and ranking metrics."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data_pipeline import FEATURE_DAYS, RETURN_END_DAY


FEATURE_GROUPS = {
    "activity volume": ["events_first_14_days"],
    "activity patterns": ["active_days_first_14_days"],
    "intent": ["cart_actions_first_14_days", "transactions_first_14_days"],
    "recency": ["days_since_last_event"],
}
# These features use information from two groups, so remove them with either one.
DERIVED_FEATURE_GROUPS = {
    "events_per_active_day": ("activity volume", "activity patterns"),
    "events_days_7_13": ("activity volume", "recency"),
    "longest_inactivity_gap_days": ("activity patterns", "recency"),
    "active_weeks_first_14_days": ("activity patterns", "recency"),
}
FEATURES = [column for columns in FEATURE_GROUPS.values() for column in columns]
FEATURES += list(DERIVED_FEATURE_GROUPS)
FEATURE_SETS = {"full": FEATURES}
for group in FEATURE_GROUPS:
    removed = set(FEATURE_GROUPS[group]) | {
        column for column, groups in DERIVED_FEATURE_GROUPS.items() if group in groups
    }
    FEATURE_SETS[f"without {group}"] = [column for column in FEATURES if column not in removed]
# The combined count is kept only for the simple intent rule.
ALL_FEATURES = FEATURES + ["high_intent_actions_first_14_days"]

TARGET = "returned_days_15_44"
VALIDATION_SCORE_START = pd.Timestamp("2015-07-03", tz="UTC")
VALIDATION_SCORE_END = pd.Timestamp("2015-07-08", tz="UTC")
HOLDOUT_SCORE_START = pd.Timestamp("2015-08-08", tz="UTC")
TOP_FRACTION = 0.1
# Direction makes every rule use higher scores for visitors ranked first.
RULES = {
    "event count": ("events_first_14_days", 1),
    "active days": ("active_days_first_14_days", 1),
    "high intent": ("high_intent_actions_first_14_days", 1),
    "recency": ("days_since_last_event", -1),
}


def rule_scores(cohort: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        name: cohort[column].to_numpy() * direction
        for name, (column, direction) in RULES.items()
    })


def validate_scores(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be nonempty vectors of equal length.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores must be finite.")
    return labels, scores


def top_fraction_metrics(labels: pd.Series, scores, fraction=TOP_FRACTION) -> tuple[float, float]:
    labels, scores = validate_scores(labels, scores)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Selection fraction must be in (0, 1].")
    top_count = max(1, int(len(labels) * fraction))
    cutoff = np.sort(scores)[-top_count]
    above = scores > cutoff
    tied = scores == cutoff
    # Average over random selection when scores tie at the cutoff.
    captured = labels[above].sum() + labels[tied].mean() * (top_count - above.sum())
    precision = captured / top_count
    recall = captured / labels.sum() if labels.sum() else 0.0
    return precision, recall


def ranking_metrics(labels: pd.Series, scores) -> dict[str, float]:
    labels, scores = validate_scores(labels, scores)
    metrics = {"average_precision": float(average_precision_score(labels, scores)) if labels.sum() else 0.0}
    for fraction in (0.05, 0.10, 0.20):
        precision, capture = top_fraction_metrics(labels, scores, fraction)
        metrics[f"capture_at_{int(fraction * 100)}pct"] = float(capture)
        if fraction == TOP_FRACTION:
            metrics["top_10_precision"] = float(precision)
    return metrics


def make_model(name: str, params: dict | None = None):
    params = params or {}
    if name == "logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **params))
    if name == "XGBoost":
        settings = {
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
            "eval_metric": "logloss", "random_state": 42, "n_jobs": 4,
        }
        return XGBClassifier(**(settings | params))
    raise ValueError(f"Unknown model: {name}")


def split_cohort(cohort: pd.DataFrame):
    """Group visitors by when their features and outcomes are available."""
    # XGBoost's histogram construction can depend on row order, even with a fixed seed.
    cohort = cohort.sort_values(["first_event_time", "user_id"]).reset_index(drop=True)
    feature_ready = cohort["first_event_time"] + pd.Timedelta(days=FEATURE_DAYS)
    labels_ready = cohort["first_event_time"] + pd.Timedelta(days=RETURN_END_DAY)
    train = cohort.loc[labels_ready <= VALIDATION_SCORE_START]
    validation = cohort.loc[
        (feature_ready >= VALIDATION_SCORE_START)
        & (feature_ready < VALIDATION_SCORE_END)
    ]
    test = cohort.loc[feature_ready >= HOLDOUT_SCORE_START]
    final_train = cohort.loc[labels_ready <= HOLDOUT_SCORE_START]
    return train, validation, test, final_train


def evaluate_candidate(train, validation, model_name, feature_set, params=None):
    """Fit on training visitors and score validation without changing either frame."""
    columns = FEATURE_SETS[feature_set]
    model = make_model(model_name, params)
    model.fit(train[columns], train[TARGET])
    predictions = model.predict_proba(validation[columns])[:, 1]
    return predictions, ranking_metrics(validation[TARGET], predictions)


def choose_ranking(metrics, candidates):
    """Choose by validation AP; insertion order breaks exact ties."""
    name = max(metrics, key=lambda key: metrics[key]["average_precision"])
    spec = candidates[name] if name in candidates else {"model": name, "feature_set": "full", "params": {}}
    return name, {**spec, "params": dict(spec["params"])}


def paired_bootstrap(labels, model_scores, baseline_scores, repeats=500, seed=42):
    """Visitor bootstrap of fixed predictions, pairing both rankings on each draw."""
    labels, model_scores = validate_scores(labels, model_scores)
    _, baseline_scores = validate_scores(labels, baseline_scores)
    if repeats < 2:
        raise ValueError("At least two bootstrap repeats are required.")
    keys = ["average_precision", "capture_at_5pct", "capture_at_10pct", "capture_at_20pct"]
    model_metrics = ranking_metrics(labels, model_scores)
    baseline_metrics = ranking_metrics(labels, baseline_scores)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        indices = rng.integers(len(labels), size=len(labels))
        first = ranking_metrics(labels[indices], model_scores[indices])
        second = ranking_metrics(labels[indices], baseline_scores[indices])
        differences.append([first[key] - second[key] for key in keys])
    bounds = np.quantile(differences, [0.025, 0.975], axis=0)
    return pd.DataFrame({
        "difference": [model_metrics[key] - baseline_metrics[key] for key in keys],
        "lower 95% bound": bounds[0], "upper 95% bound": bounds[1],
    }, index=keys)
