"""Build a retention cohort measured from each user's first observed event."""

from pathlib import Path

import duckdb


PROJECT_DIR = Path(__file__).resolve().parent
RAW_EVENTS_PATH = PROJECT_DIR / "data" / "raw" / "events.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "retention_cohort.parquet"

ACTIVATION_DAYS = 3
MIN_ACTIVATION_EVENTS = 3
MAX_ACTIVATION_EVENTS = 30
RETURN_START_DAY = 27
RETURN_END_DAY = 34  # exclusive


def run_pipeline(
    raw_events_path: Path = RAW_EVENTS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    raw_events_path = Path(raw_events_path)
    output_path = Path(output_path)

    if not raw_events_path.is_file():
        raise FileNotFoundError(f"Events file not found: {raw_events_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as con:
        con.execute(
            """
            CREATE TABLE events AS
            SELECT
                visitorid AS user_id,
                to_timestamp(CAST(timestamp AS BIGINT) / 1000.0) AS event_time,
                event AS event_type
            FROM read_csv_auto(?)
            """,
            [str(raw_events_path)],
        )

        row_count, missing_count = con.execute(
            """
            SELECT COUNT(*), COUNT(*) FILTER (
                WHERE user_id IS NULL OR event_time IS NULL OR event_type IS NULL
            )
            FROM events
            """
        ).fetchone()
        if row_count == 0:
            raise ValueError("Events file is empty.")
        if missing_count:
            raise ValueError(f"Events file has {missing_count} rows with missing required values.")

        con.execute(
            f"""
            COPY (
                WITH first_event AS (
                    SELECT user_id, MIN(event_time) AS first_event_time
                    FROM events
                    GROUP BY user_id
                ),
                user_activity AS (
                    SELECT
                        f.user_id,
                        f.first_event_time,
                        COUNT(e.event_type) AS activation_events,
                        COUNT(*) FILTER (
                            WHERE e.event_type IN ('addtocart', 'transaction')
                        ) AS high_intent_actions
                    FROM first_event f
                    LEFT JOIN events e
                        ON e.user_id = f.user_id
                        AND e.event_time >= f.first_event_time
                        AND e.event_time < f.first_event_time + INTERVAL '{ACTIVATION_DAYS} days'
                    GROUP BY f.user_id, f.first_event_time
                ),
                eligible_users AS (
                    SELECT a.*
                    FROM user_activity a
                    CROSS JOIN (SELECT MAX(event_time) AS latest_event FROM events) d
                    WHERE a.first_event_time + INTERVAL '{RETURN_END_DAY} days' <= d.latest_event
                      AND a.activation_events >= {MIN_ACTIVATION_EVENTS}
                      AND a.activation_events <= {MAX_ACTIVATION_EVENTS}
                ),
                labeled_users AS (
                    SELECT
                        u.*,
                        CASE WHEN u.high_intent_actions > 0 THEN 1 ELSE 0 END
                            AS completed_high_intent_action,
                        CASE WHEN COUNT(*) FILTER (
                            WHERE e.event_time >= u.first_event_time + INTERVAL '{RETURN_START_DAY} days'
                              AND e.event_time < u.first_event_time + INTERVAL '{RETURN_END_DAY} days'
                        ) > 0 THEN 1 ELSE 0 END AS returned_day30_window
                    FROM eligible_users u
                    LEFT JOIN events e ON e.user_id = u.user_id
                    GROUP BY u.user_id, u.first_event_time, u.activation_events,
                             u.high_intent_actions
                )
                SELECT * FROM labeled_users
            ) TO ? (FORMAT PARQUET)
            """,
            [str(output_path)],
        )

        summary = con.execute(
            """
            SELECT
                completed_high_intent_action,
                COUNT(*) AS users,
                SUM(returned_day30_window) AS returned_users,
                ROUND(AVG(returned_day30_window) * 100, 2) AS return_rate_pct
            FROM read_parquet(?)
            GROUP BY completed_high_intent_action
            ORDER BY completed_high_intent_action
            """,
            [str(output_path)],
        ).fetchall()

    cohort_size = sum(row[1] for row in summary)
    print(f"Saved {cohort_size:,} users to {output_path}")
    if not summary:
        print("No eligible users were found, so there are no return rates to show.")
        return

    print("\nDay-30 return rates:")
    for completed_action, users, returned_users, return_rate in summary:
        action_label = "Yes" if completed_action else "No"
        print(f"High-intent action {action_label}: {return_rate:.2f}% ({returned_users:,}/{users:,} users)")


if __name__ == "__main__":
    run_pipeline()
