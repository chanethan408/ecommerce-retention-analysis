import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_pipeline import run_pipeline


def _write_events(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["visitorid", "timestamp", "event", "itemid"])
        writer.writerows(rows)


def _event(start, visitor_id, day, event_type="view", second=0):
    event_time = start + timedelta(days=day, seconds=second)
    timestamp_ms = int(event_time.timestamp() * 1000)
    return [visitor_id, timestamp_ms, event_type, "item-1"]


def _run_pipeline(tmp_path, rows):
    raw_path = tmp_path / "events.csv"
    output_path = tmp_path / "processed" / "cohort.parquet"
    _write_events(raw_path, rows)
    run_pipeline(raw_path, output_path)
    return pd.read_parquet(output_path)


def test_pipeline_labels_high_intent_and_return_window(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        _event(start, "high_intent", 0),
        _event(start, "high_intent", 1, "addtocart"),
        _event(start, "high_intent", 2),
        _event(start, "high_intent", 27),
        _event(start, "day_33_return", 0),
        _event(start, "day_33_return", 1),
        _event(start, "day_33_return", 2),
        _event(start, "day_33_return", 33, second=86399),
        _event(start, "later_start_transaction", 10),
        _event(start, "later_start_transaction", 11, "transaction"),
        _event(start, "later_start_transaction", 12),
        _event(start, "later_start_transaction", 37),
        _event(start, "no_return", 0),
        _event(start, "no_return", 1),
        _event(start, "no_return", 2),
        _event(start, "late_action", 0),
        _event(start, "late_action", 1),
        _event(start, "late_action", 2),
        _event(start, "late_action", 5, "addtocart"),
        _event(start, "late_action", 27),
        _event(start, "day_34_boundary", 0),
        _event(start, "day_34_boundary", 1),
        _event(start, "day_34_boundary", 2),
        _event(start, "day_34_boundary", 34),
        _event(start, "latest_event", 60),
    ]

    cohort = _run_pipeline(tmp_path, rows).set_index("user_id")

    assert cohort.index.is_unique
    assert set(cohort.index) == {
        "high_intent",
        "day_33_return",
        "later_start_transaction",
        "no_return",
        "late_action",
        "day_34_boundary",
    }
    assert cohort.loc["high_intent", "completed_high_intent_action"] == 1
    assert cohort.loc["later_start_transaction", "completed_high_intent_action"] == 1
    assert cohort.loc["no_return", "completed_high_intent_action"] == 0
    assert cohort.loc["late_action", "completed_high_intent_action"] == 0
    # The return window includes Days 27-33 and excludes Day 34.
    assert cohort["returned_day30_window"].to_dict() == {
        "high_intent": 1,
        "day_33_return": 1,
        "later_start_transaction": 1,
        "no_return": 0,
        "late_action": 1,
        "day_34_boundary": 0,
    }


def test_pipeline_excludes_users_outside_cohort_rules(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        _event(start, "too_few", 0),
        _event(start, "too_few", 1),
        # This user's first event is Day 30, leaving fewer than 34 days of follow-up.
        _event(start, "late_start", 30),
        _event(start, "late_start", 31),
        _event(start, "late_start", 32),
        _event(start, "latest_event", 60),
    ]
    rows.extend(_event(start, "max_allowed", 0, second=i) for i in range(30))
    rows.extend(_event(start, "too_many", 0, second=i) for i in range(31))

    cohort = _run_pipeline(tmp_path, rows).set_index("user_id")

    assert cohort.index.is_unique
    assert cohort.index.tolist() == ["max_allowed"]
    assert cohort.loc["max_allowed", "activation_events"] == 30


def test_pipeline_output_is_unchanged_on_rerun(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        _event(start, "user", 0),
        _event(start, "user", 1),
        _event(start, "user", 2),
        _event(start, "user", 30),
        _event(start, "latest_event", 60),
    ]
    raw_path = tmp_path / "events.csv"
    output_path = tmp_path / "processed" / "cohort.parquet"
    _write_events(raw_path, rows)

    run_pipeline(raw_path, output_path)
    first_output = pd.read_parquet(output_path)
    run_pipeline(raw_path, output_path)
    second_output = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(first_output, second_output)


def test_pipeline_requires_input_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Events file not found"):
        run_pipeline(tmp_path / "missing.csv", tmp_path / "cohort.parquet")


def test_pipeline_rejects_rows_with_missing_required_values(tmp_path):
    raw_path = tmp_path / "events.csv"
    _write_events(
        raw_path,
        [
            ["1", 1735689600000, "view", "item-1"],
            ["", "", "view", "item-2"],
        ],
    )

    with pytest.raises(ValueError, match="missing required values"):
        run_pipeline(raw_path, tmp_path / "cohort.parquet")


def test_pipeline_rejects_empty_input(tmp_path):
    raw_path = tmp_path / "events.csv"
    _write_events(raw_path, [])

    with pytest.raises(ValueError, match="Events file is empty"):
        run_pipeline(raw_path, tmp_path / "cohort.parquet")
