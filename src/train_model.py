"""Train and compare return-ranking baselines."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if __package__:
    from .data_pipeline import FEATURE_DAYS, RETURN_END_DAY
else:
    from data_pipeline import FEATURE_DAYS, RETURN_END_DAY


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "retention_cohort.parquet"

FEATURES = [
    "events_first_14_days",
    "active_days_first_14_days",
    "high_intent_actions_first_14_days",
]
TARGET = "returned_days_15_44"
VALIDATION_SCORE_START = pd.Timestamp("2015-07-03", tz="UTC")
VALIDATION_SCORE_END = pd.Timestamp("2015-07-08", tz="UTC")
HOLDOUT_SCORE_START = pd.Timestamp("2015-08-08", tz="UTC")
TOP_FRACTION = 0.1  # Illustrative until campaign capacity is known.
RULES = {
    "event count": "events_first_14_days",
    "active days": "active_days_first_14_days",
}


def _top_fraction_metrics(labels: pd.Series, scores) -> tuple[float, float]:
    top_count = max(1, int(len(labels) * TOP_FRACTION))
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    cutoff = np.sort(scores)[-top_count]
    above = scores > cutoff
    tied = scores == cutoff
    # Expected outcome of random selection within a tied boundary group.
    captured = labels[above].sum() + labels[tied].mean() * (top_count - above.sum())
    precision = captured / top_count
    recall = captured / labels.sum() if labels.sum() else 0.0
    return precision, recall


def _ranking_metrics(labels: pd.Series, scores) -> dict[str, float]:
    precision, recall = _top_fraction_metrics(labels, scores)
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "top_10_precision": float(precision),
        "top_10_recall": float(recall),
    }


def _make_model(name: str):
    if name == "logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )


def _split_cohort(cohort: pd.DataFrame):
    """Group visitors by when their features and outcomes are available."""
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


def train_model(data_path: Path = DATA_PATH) -> dict[str, object]:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Cohort file not found: {data_path}. Run src/data_pipeline.py first."
        )

    cohort = pd.read_parquet(data_path)
    required_columns = ["first_event_time", *FEATURES, TARGET]
    missing_columns = [column for column in required_columns if column not in cohort]
    if missing_columns:
        raise ValueError(f"Cohort is missing required columns: {', '.join(missing_columns)}")
    if len(cohort) < 2:
        raise ValueError("At least two users are needed for a train/test split.")

    cohort = cohort.sort_values("first_event_time").reset_index(drop=True)
    train, validation, test, final_train = _split_cohort(cohort)

    if train.empty or train[TARGET].nunique() < 2:
        raise ValueError("Training data before the validation cutoff needs both outcomes.")
    if validation.empty or validation[TARGET].nunique() < 2:
        raise ValueError("The fixed validation period needs both outcome classes.")
    if test.empty or test[TARGET].nunique() < 2:
        raise ValueError("The fixed holdout period needs both outcome classes.")

    validation_labels_mature = (
        VALIDATION_SCORE_END + pd.Timedelta(days=RETURN_END_DAY - FEATURE_DAYS)
    )
    validation_scores = {
        name: average_precision_score(validation[TARGET], validation[column])
        for name, column in RULES.items()
    }
    for name in ("logistic regression", "XGBoost"):
        candidate = _make_model(name)
        candidate.fit(train[FEATURES], train[TARGET])
        predictions = candidate.predict_proba(validation[FEATURES])[:, 1]
        validation_scores[name] = average_precision_score(
            validation[TARGET], predictions
        )

    selected_name = max(validation_scores, key=validation_scores.get)

    if final_train.empty or final_train[TARGET].nunique() < 2:
        raise ValueError("The final training period needs both outcome classes.")

    if selected_name in RULES:
        test_predictions = test[RULES[selected_name]].to_numpy()
    else:
        final_model = _make_model(selected_name)
        final_model.fit(final_train[FEATURES], final_train[TARGET])
        test_predictions = final_model.predict_proba(test[FEATURES])[:, 1]
    test_baseline = float(test[TARGET].mean())
    rule_results = {
        name: _ranking_metrics(test[TARGET], test[column])
        for name, column in RULES.items()
    }
    selected_results = (
        rule_results[selected_name] if selected_name in RULES
        else _ranking_metrics(test[TARGET], test_predictions)
    )

    print(f"Training users for comparison: {len(train):,}")
    print(f"Validation users: {len(validation):,}")
    print(f"Validation share of eligible cohort: {len(validation) / len(cohort):.1%}")
    print(
        "Validation scoring period (UTC): "
        f"{VALIDATION_SCORE_START} to {VALIDATION_SCORE_END}"
    )
    print(f"Training labels observable by (UTC): {VALIDATION_SCORE_START}")
    print(f"Validation outcomes fully observable by (UTC): {validation_labels_mature}")
    print(f"Validation return rate: {validation[TARGET].mean():.3f}")
    for name, score in validation_scores.items():
        print(f"{name} validation AP: {score:.5f}")
    print(f"Selected ranking: {selected_name}")
    print(f"Final training users: {len(final_train):,}")
    print(
        "Final holdout feature-ready period (UTC): "
        f"{test['first_event_time'].min() + pd.Timedelta(days=FEATURE_DAYS)} to "
        f"{test.iloc[-1]['first_event_time'] + pd.Timedelta(days=FEATURE_DAYS)}"
    )
    print(f"Final training labels observable by (UTC): {HOLDOUT_SCORE_START}")
    print(f"Final holdout users: {len(test):,}")
    print(f"Final holdout return rate (random-ranking AP baseline): {test_baseline:.3f}")
    for name, metrics in {selected_name: selected_results, **rule_results}.items():
        print(
            f"{name} holdout: AP {metrics['average_precision']:.3f}, "
            f"top-10% precision {metrics['top_10_precision']:.1%}, "
            f"recall {metrics['top_10_recall']:.1%}"
        )

    return {
        "train_users": len(train),
        "validation_users": len(validation),
        "final_train_users": len(final_train),
        "test_users": len(test),
        "validation_baseline_average_precision": float(validation[TARGET].mean()),
        "validation_scores": {name: float(score) for name, score in validation_scores.items()},
        "selected_model": selected_name,
        "test_baseline_average_precision": test_baseline,
        "test_average_precision": selected_results["average_precision"],
        "test_top_10_precision": selected_results["top_10_precision"],
        "test_top_10_recall": selected_results["top_10_recall"],
        "test_rules": rule_results,
    }


if __name__ == "__main__":
    train_model()
