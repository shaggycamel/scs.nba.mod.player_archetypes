"""Run the weekly production job and record its status.

Usage:
    uv run --group data python -m scripts.job
    uv run --group data python -m scripts.job --cadence weekly

The job is deliberately sequential: the publisher only runs if the live
pipeline succeeds. One row is written to ``util.update_log`` at the end, so a
cron failure is visible from the database even when cron has no mail agent.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime

import polars as pl

SOURCE = "scs.nba.mod.player_archetypes"
TASK = "player_archetypes.job"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", default="weekly")
    parser.add_argument("--db-con", default="cockroach-write")
    parser.add_argument("--source", default=os.environ.get("UPDATE_LOG_SOURCE", SOURCE))
    return parser.parse_args(argv)


def log_run(
    db,
    kickoff: datetime,
    cadence: str,
    source: str,
    success: bool,
    error: str | None,
    row_count: int | None,
) -> None:
    frame = pl.DataFrame(
        {
            "source": source,
            "job_kickoff": kickoff,
            "cadence": cadence,
            "task": TASK,
            "task_time": datetime.now(UTC),
            "success": success,
            "error_message": error,
            "row_count": row_count,
        },
        schema_overrides={"row_count": pl.Int64},
    )
    db.write(frame, "update_log", "util")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kickoff = datetime.now(UTC)

    commands = [
        (
            "pipeline",
            [
                sys.executable,
                "-m",
                "scripts.pipeline",
                "--regular-season-only",
                "--current-season-only",
            ],
        ),
        ("publish", [sys.executable, "-m", "scripts.publish"]),
    ]

    error = None
    for name, command in commands:
        log("running: " + " ".join(command[1:]))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            error = f"{name} exited with status {result.returncode}"
            break

    try:
        from scs_hub import SportsHub
    except ImportError:
        print(
            "error: scripts.job requires scs-hub; run with "
            "`uv run --group data python -m scripts.job`",
            flush=True,
        )
        return 1

    try:
        db = SportsHub(db_con=args.db_con).db
    except Exception as exc:  # noqa: BLE001 - status logging must not mask this
        print(f"error: could not connect to [{args.db_con}]: {exc}", flush=True)
        return 1

    row_count = None
    if error is None:
        try:
            row_count = db.read("SELECT count(*) AS n FROM mod.player_role")[0, "n"]
        except Exception as exc:  # noqa: BLE001 - still write the run status
            print(f"warning: could not count mod.player_role: {exc}", flush=True)

    try:
        log_run(
            db,
            kickoff=kickoff,
            cadence=args.cadence,
            source=args.source,
            success=error is None,
            error=error,
            row_count=row_count,
        )
    except Exception as exc:  # noqa: BLE001 - the job result is more important
        print(f"error: could not write util.update_log: {exc}", flush=True)
        return 1

    if error is not None:
        log(f"failed: {error}")
        return 1
    if row_count is None:
        log("completed; row count unavailable")
    else:
        log(f"completed; mod.player_role has {row_count:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
