import pandas as pd
import numpy as np
import pytest

from src.ranking import (
    FEATURE_DAYS,
    HOLDOUT_SCORE_START,
    RETURN_END_DAY,
    VALIDATION_SCORE_END,
    ranking_metrics,
    rule_scores,
    split_cohort,
    top_fraction_metrics,
    choose_ranking,
    evaluate_candidate,
    paired_bootstrap,
    FEATURE_SETS,
    TARGET,
)


def test_top_fraction_averages_boundary_ties_without_using_row_order():
    labels = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    assert top_fraction_metrics(labels, scores) == pytest.approx((0.5, 0.5))
    assert top_fraction_metrics(labels.iloc[::-1], scores[::-1]) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize("first_seen, expected_groups", [
    ("2015-05-19 00:00:00", {"train", "final_train"}),
    ("2015-05-19 00:00:01", {"final_train"}),
    ("2015-06-18 23:59:59", {"final_train"}),
    ("2015-06-19 00:00:00", {"validation", "final_train"}),
    ("2015-06-23 23:59:59", {"validation", "final_train"}),
    ("2015-06-24 00:00:00", {"final_train"}),
    ("2015-06-24 00:00:01", set()),
    ("2015-07-24 23:59:59", set()),
    ("2015-07-25 00:00:00", {"test"}),
])
def test_visitors_go_into_the_right_date_groups(first_seen, expected_groups):
    cohort = pd.DataFrame({"user_id": [1], "first_event_time": [pd.Timestamp(first_seen, tz="UTC")]})
    groups = split_cohort(cohort)
    names = ("train", "validation", "test", "final_train")
    actual_groups = {name for name, rows in zip(names, groups) if not rows.empty}
    assert actual_groups == expected_groups


def test_shuffled_cohort_produces_identical_order_in_every_split():
    cohort = pd.DataFrame({
        "user_id": [8, 2, 7, 1, 6, 3, 5, 4],
        "first_event_time": pd.to_datetime([
            "2015-05-18", "2015-05-18", "2015-06-19", "2015-06-19",
            "2015-06-24", "2015-06-24", "2015-07-25", "2015-07-25",
        ], utc=True),
        TARGET: [0, 1] * 4,
    })
    original = cohort.copy(deep=True)
    expected = ([2, 8], [1, 7], [4, 5], [2, 8, 1, 7, 3, 6])
    first = split_cohort(cohort)
    shuffled = split_cohort(cohort.sample(frac=1, random_state=42))
    for left, right, ids in zip(first, shuffled, expected):
        pd.testing.assert_frame_equal(left, right)
        assert left["user_id"].tolist() == ids
    pd.testing.assert_frame_equal(cohort, original)


def test_validation_outcomes_are_ready_before_holdout_scoring():
    outcomes_ready = VALIDATION_SCORE_END + pd.Timedelta(days=RETURN_END_DAY - FEATURE_DAYS)
    assert outcomes_ready <= HOLDOUT_SCORE_START


def test_recency_rule_puts_recent_visitors_first():
    cohort = pd.DataFrame({
        "events_first_14_days": [100, 1],
        "active_days_first_14_days": [5, 1],
        "high_intent_actions_first_14_days": [0, 1],
        "days_since_last_event": [10.0, 0.5],
    })
    scores = rule_scores(cohort)
    assert scores["recency"].idxmax() == 1
    assert scores["event count"].idxmax() == 0
    assert scores["high intent"].idxmax() == 1


def test_capture_reports_share_of_returners_at_each_selection_size():
    labels = pd.Series([1, 1] + [0] * 17 + [1])
    metrics = ranking_metrics(labels, list(range(20, 0, -1)))
    assert metrics["capture_at_5pct"] == pytest.approx(1 / 3)
    assert metrics["capture_at_10pct"] == pytest.approx(2 / 3)
    assert metrics["capture_at_20pct"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("labels,scores", [
    ([], []), ([1], [1, 2]), ([0, 2], [1, 2]), ([0, 1], [0, np.nan]),
    ([0, 1], [0, np.inf]), ([[0, 1]], [[0, 1]]),
])
def test_metrics_reject_invalid_vectors(labels, scores):
    with pytest.raises(ValueError):
        ranking_metrics(labels, scores)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, np.nan])
def test_metrics_reject_invalid_capacity(fraction):
    with pytest.raises(ValueError, match="fraction"):
        top_fraction_metrics([0, 1], [1, 2], fraction)


def test_no_returners_has_zero_ap_and_capture():
    assert set(ranking_metrics([0, 0], [2, 1]).values()) == {0}


def test_selection_allows_rules_and_preserves_candidate_configuration():
    spec = {"model": "XGBoost", "feature_set": "without recency", "params": {"max_depth": 2}}
    metrics = {"recency": {"average_precision": 0.2}, "trial": {"average_precision": 0.3}}
    name, chosen = choose_ranking(metrics, {"trial": spec})
    assert name == "trial" and chosen == spec
    chosen["params"]["max_depth"] = 9
    assert spec["params"] == {"max_depth": 2}
    metrics["recency"]["average_precision"] = 0.4
    assert choose_ranking(metrics, {"trial": spec}) == (
        "recency", {"model": "recency", "feature_set": "full", "params": {}},
    )


def test_candidate_fits_selected_features_on_training_only(monkeypatch):
    from src import ranking

    columns = FEATURE_SETS["without recency"]
    train = pd.DataFrame({**{c: [1, 2] for c in columns}, TARGET: [0, 1]})
    validation = pd.DataFrame({**{c: [3, 4] for c in columns}, TARGET: [1, 0]})

    class Model:
        def fit(self, features, labels):
            pd.testing.assert_frame_equal(features, train[columns])
            pd.testing.assert_series_equal(labels, train[TARGET])

        def predict_proba(self, features):
            pd.testing.assert_frame_equal(features, validation[columns])
            return np.array([[0.2, 0.8], [0.9, 0.1]])

    monkeypatch.setattr(ranking, "make_model", lambda name, params: Model())
    predictions, metrics = evaluate_candidate(train, validation, "XGBoost", "without recency")
    assert predictions.tolist() == [0.8, 0.1]
    assert metrics["average_precision"] == 1


def test_paired_bootstrap_identical_rankings_have_exactly_zero_difference():
    result = paired_bootstrap([0, 1] * 10, [0.1, 0.9] * 10, [0.1, 0.9] * 10, repeats=20)
    assert (result.to_numpy() == 0).all()
