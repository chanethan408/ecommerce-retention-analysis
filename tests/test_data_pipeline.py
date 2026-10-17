import csv
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

from src.data_pipeline import run_pipeline


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


def test_pipeline_builds_day14_features_and_days15_to44_label(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        _event(start, "returner", 0),
        _event(start, "returner", 1, "addtocart"),
        _event(start, "returner", 13, "transaction"),
        _event(start, "returner", 14),
        _event(start, "returner", 15),
        _event(start, "returner", 44),
        _event(start, "returner", 45),
        _event(start, "no_return", 0),
        _event(start, "no_return", 2),
        _event(start, "day_45_only", 0),
        _event(start, "day_45_only", 45),
        _event(start, "late_start", 30),
        _event(start, "late_start", 31),
        _event(start, "late_start", 32),
        _event(start, "latest_event", 70),
    ]

    cohort = _run_pipeline(tmp_path, rows).set_index("user_id")

    assert cohort.index.is_unique
    assert set(cohort.index) == {"returner", "no_return", "day_45_only"}
    assert cohort.loc["returner", "events_first_14_days"] == 3
    assert cohort.loc["returner", "active_days_first_14_days"] == 3
    assert cohort.loc["returner", "high_intent_actions_first_14_days"] == 2
    assert cohort["returned_days_15_44"].to_dict() == {
        "returner": 1,
        "no_return": 0,
        "day_45_only": 0,
    }


def test_pipeline_output_is_unchanged_on_rerun(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        _event(start, "user", 0),
        _event(start, "user", 20),
        _event(start, "another_user", 1),
        _event(start, "latest_event", 60),
    ]
    raw_path = tmp_path / "events.csv"
    output_path = tmp_path / "processed" / "cohort.parquet"
    _write_events(raw_path, rows)

    run_pipeline(raw_path, output_path)
    first_output = pd.read_parquet(output_path)
    run_pipeline(raw_path, output_path)
    second_output = pd.read_parquet(output_path)

    # Compare the same visitors; row order can change between runs.
    first_output = first_output.sort_values("user_id").reset_index(drop=True)
    second_output = second_output.sort_values("user_id").reset_index(drop=True)
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


def test_pipeline_uses_utc_dates_even_when_connection_defaults_to_pacific(tmp_path, monkeypatch):
    connect = duckdb.connect

    def pacific_connection(*args, **kwargs):
        con = connect(*args, **kwargs)
        con.execute("SET TimeZone = 'America/Los_Angeles'")
        return con

    monkeypatch.setattr(duckdb, "connect", pacific_connection)
    start = datetime(2025, 1, 1, 23, 30, tzinfo=timezone.utc)
    cohort = _run_pipeline(tmp_path, [
        _event(start, "user", 0),
        _event(start, "user", 0, second=3600),
        _event(start, "latest", 60),
    ]).set_index("user_id")
    assert cohort.loc["user", "active_days_first_14_days"] == 2


def test_events_in_scoring_gap_are_neither_features_nor_returns(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cohort = _run_pipeline(tmp_path, [
        _event(start, "user", 0),
        _event(start, "user", 14),
        _event(start, "user", 14, second=86399),
        _event(start, "latest", 60),
    ]).set_index("user_id")
    assert cohort.loc["user", "events_first_14_days"] == 1
    assert cohort.loc["user", "returned_days_15_44"] == 0


def test_new_features_describe_when_activity_happened(tmp_path):
    start = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    cohort = _run_pipeline(tmp_path, [
        _event(start, "spread", 0),
        _event(start, "spread", 6, "addtocart", second=86399),
        _event(start, "spread", 7, "addtocart"),
        _event(start, "spread", 13, "transaction"),
        _event(start, "spread", 14, "transaction"),
        _event(start, "burst", 0),
        _event(start, "burst", 0, "addtocart"),
        _event(start, "latest", 60),
    ]).set_index("user_id")

    spread = cohort.loc["spread"]
    assert spread["days_since_last_event"] == pytest.approx(1)
    assert spread["events_days_7_13"] == 2
    assert spread["events_per_active_day"] == pytest.approx(4 / 3)
    assert spread["longest_inactivity_gap_days"] == pytest.approx(7 - 1 / 86400)
    assert spread["active_weeks_first_14_days"] == 2
    assert spread["cart_actions_first_14_days"] == 2
    assert spread["transactions_first_14_days"] == 1

    burst = cohort.loc["burst"]
    assert burst["days_since_last_event"] == pytest.approx(14)
    assert burst["events_days_7_13"] == 0
    assert burst["events_per_active_day"] == pytest.approx(2)
    assert burst["longest_inactivity_gap_days"] == pytest.approx(14)
    assert burst["active_weeks_first_14_days"] == 1
    assert burst["transactions_first_14_days"] == 0


def test_later_events_do_not_change_features(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        _event(start, "user", 0),
        _event(start, "user", 13, "addtocart", second=86399),
        _event(start, "latest", 60),
    ]
    before = _run_pipeline(tmp_path, rows).set_index("user_id")
    rows += [
        _event(start, "user", 14, "transaction"),
        _event(start, "user", 15, "transaction"),
        _event(start, "user", 30, "addtocart"),
    ]
    after = _run_pipeline(tmp_path, rows).set_index("user_id")
    feature_columns = before.columns.drop(["first_event_time", "returned_days_15_44"])
    pd.testing.assert_series_equal(
        before.loc["user", feature_columns], after.loc["user", feature_columns],
    )
    assert before.loc["user", "returned_days_15_44"] == 0
    assert after.loc["user", "returned_days_15_44"] == 1
