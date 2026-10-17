"""Evaluate a ranking chosen using validation data."""

import pandas as pd

from .train_model import (
    FEATURE_DAYS,
    FEATURE_SETS,
    HOLDOUT_SCORE_START,
    RETURN_END_DAY,
    RULES,
    TARGET,
    VALIDATION_SCORE_END,
    _make_model,
    _ranking_metrics,
    _rule_scores,
    _split_cohort,
)


def evaluate_holdout(
    cohort: pd.DataFrame, selected_name: str, feature_set: str = "full",
    params: dict | None = None,
) -> dict[str, object]:
    """Refit the chosen model and compare it with activity rules on the holdout."""
    if selected_name not in {*RULES, "logistic regression", "XGBoost"}:
        raise ValueError(f"Unknown ranking: {selected_name}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    features = FEATURE_SETS[feature_set]
    outcomes_ready = VALIDATION_SCORE_END + pd.Timedelta(days=RETURN_END_DAY - FEATURE_DAYS)
    if outcomes_ready > HOLDOUT_SCORE_START:
        raise ValueError("Validation outcomes must be ready before holdout scoring.")

    cohort = cohort.sort_values("first_event_time").reset_index(drop=True)
    _, _, holdout, final_train = _split_cohort(cohort)
    if final_train.empty or final_train[TARGET].nunique() < 2:
        raise ValueError("Final training data needs both outcomes.")
    if holdout.empty or holdout[TARGET].nunique() < 2:
        raise ValueError("Holdout data needs both outcomes.")

    scores = _rule_scores(holdout)
    scores.insert(0, "returned", holdout[TARGET].to_numpy())
    if selected_name not in RULES:
        model = _make_model(selected_name, params)
        model.fit(final_train[features], final_train[TARGET])
        scores[f"{selected_name} / {feature_set}"] = model.predict_proba(holdout[features])[:, 1]

    metrics = {
        name: _ranking_metrics(scores["returned"], scores[name])
        for name in scores.columns.drop("returned")
    }
    return {
        "selected_model": selected_name,
        "feature_set": feature_set,
        "params": dict(params or {}),
        "final_train_users": len(final_train),
        "test_users": len(holdout),
        "return_rate": float(scores["returned"].mean()),
        "metrics": pd.DataFrame(metrics).T,
        "holdout_scores": scores,
    }
