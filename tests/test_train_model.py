import pandas as pd
import pytest

from src.train_model import (
    _split_cohort,
    _top_fraction_metrics,
    VALIDATION_SCORE_END,
    HOLDOUT_SCORE_START,
    FEATURE_DAYS,
    RETURN_END_DAY,
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
    # Dates just before, at, and after the limits should land in the right groups.
    cohort = pd.DataFrame({"first_event_time": [pd.Timestamp(first_seen, tz="UTC")]})
    groups = _split_cohort(cohort)
    names = ("train", "validation", "test", "final_train")
    actual_groups = {name for name, rows in zip(names, groups) if not rows.empty}
    assert actual_groups == expected_groups


def test_validation_outcomes_are_ready_before_holdout_scoring():
    # Even the end of the validation period leaves enough time to see returns.
    outcomes_ready = VALIDATION_SCORE_END + pd.Timedelta(days=RETURN_END_DAY - FEATURE_DAYS)
    assert outcomes_ready <= HOLDOUT_SCORE_START
