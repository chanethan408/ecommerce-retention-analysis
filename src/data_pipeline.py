"""Build user-level features for a day-14 return prediction."""

from pathlib import Path

import duckdb


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_EVENTS_PATH = PROJECT_DIR / "data" / "raw" / "events.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "retention_cohort.parquet"

FEATURE_DAYS = 14
RETURN_START_DAY = 15
RETURN_END_DAY = 45  # exclusive


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
        con.execute("SET TimeZone = 'UTC'")
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
            raise ValueError(
                f"Events file has {missing_count} rows with missing required values."
            )

        con.execute(
            f"""
            COPY (
                WITH first_event AS (
                    SELECT user_id, MIN(event_time) AS first_event_time
                    FROM events
                    GROUP BY user_id
                ),
                user_features AS (
                    SELECT
                        f.user_id,
                        f.first_event_time,
                        COUNT(e.event_type) AS events_first_14_days,
                        COUNT(DISTINCT CAST(e.event_time AS DATE))
                            AS active_days_first_14_days,
                        COUNT(*) FILTER (
                            WHERE e.event_type IN ('addtocart', 'transaction')
                        ) AS high_intent_actions_first_14_days
                    FROM first_event f
                    LEFT JOIN events e
                        ON e.user_id = f.user_id
                        AND e.event_time >= f.first_event_time
                        AND e.event_time < f.first_event_time
                            + INTERVAL '{FEATURE_DAYS} days'
                    GROUP BY f.user_id, f.first_event_time
                ),
                eligible_users AS (
                    SELECT u.*
                    FROM user_features u
                    CROSS JOIN (SELECT MAX(event_time) AS latest_event FROM events) d
                    WHERE u.first_event_time + INTERVAL '{RETURN_END_DAY} days'
                        <= d.latest_event
                ),
                labeled_users AS (
                    SELECT
                        u.*,
                        CASE WHEN COUNT(e.event_type) > 0 THEN 1 ELSE 0 END
                            AS returned_days_15_44
                    FROM eligible_users u
                    LEFT JOIN events e
                        ON e.user_id = u.user_id
                        AND e.event_time >= u.first_event_time
                            + INTERVAL '{RETURN_START_DAY} days'
                        AND e.event_time < u.first_event_time
                            + INTERVAL '{RETURN_END_DAY} days'
                    GROUP BY
                        u.user_id,
                        u.first_event_time,
                        u.events_first_14_days,
                        u.active_days_first_14_days,
                        u.high_intent_actions_first_14_days
                )
                SELECT * FROM labeled_users
            ) TO ? (FORMAT PARQUET)
            """,
            [str(output_path)],
        )

        user_count, returners = con.execute(
            """
            SELECT COUNT(*), SUM(returned_days_15_44)
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

    display_path = output_path.resolve()
    if display_path.is_relative_to(PROJECT_DIR):
        display_path = display_path.relative_to(PROJECT_DIR)
    else:
        display_path = Path(display_path.name)
    print(f"Saved {user_count:,} eligible users to {display_path.as_posix()}")
    if user_count:
        print(
            f"Returned during days 15-44: {returners:,} "
            f"({returners / user_count:.1%})"
        )
    else:
        print("No users had enough follow-up for the return window.")


if __name__ == "__main__":
    run_pipeline()
