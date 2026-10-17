import pandas as pd
import pytest

from src.train_model import (
    FEATURE_DAYS,
    HOLDOUT_SCORE_START,
    RETURN_END_DAY,
    VALIDATION_SCORE_END,
    _ranking_metrics,
    _rule_scores,
    _split_cohort,
    _top_fraction_metrics,
)


def test_top_fraction_averages_boundary_ties_without_using_row_order():
    labels = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    assert _top_fraction_metrics(labels, scores) == pytest.approx((0.5, 0.5))
    assert _top_fraction_metrics(labels.iloc[::-1], scores[::-1]) == pytest.approx((0.5, 0.5))


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
    cohort = pd.DataFrame({"first_event_time": [pd.Timestamp(first_seen, tz="UTC")]})
    groups = _split_cohort(cohort)
    names = ("train", "validation", "test", "final_train")
    actual_groups = {name for name, rows in zip(names, groups) if not rows.empty}
    assert actual_groups == expected_groups


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
    scores = _rule_scores(cohort)
    assert scores["recency"].idxmax() == 1
    assert scores["event count"].idxmax() == 0
    assert scores["high intent"].idxmax() == 1


def test_capture_reports_share_of_returners_at_each_selection_size():
    labels = pd.Series([1, 1] + [0] * 17 + [1])
    metrics = _ranking_metrics(labels, list(range(20, 0, -1)))
    assert metrics["capture_at_5pct"] == pytest.approx(1 / 3)
    assert metrics["capture_at_10pct"] == pytest.approx(2 / 3)
    assert metrics["capture_at_20pct"] == pytest.approx(2 / 3)
