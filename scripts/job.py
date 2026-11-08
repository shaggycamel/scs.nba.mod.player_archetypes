"""Run the weekly production job and record its status.

Usage:
    uv run --group data python -m scripts.job
    uv run --group data python -m scripts.job --cadence weekly

The job is deliberately sequential: the publisher only runs if the live
pipeline succeeds. When ``SCS_RESULT_FILE`` is set, the job writes the row
count there and lets the caller own ``util.update_log``. Otherwise it writes
its own status row, so a standalone cron failure is still visible in the
database.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

SOURCE = "scs.nba.mod.player_archetypes"
TASK = "player_archetypes.job"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", default=os.environ.get("SCS_CADENCE", "weekly"))
    parser.add_argument("--db-con", default="cockroach-write")
    parser.add_argument("--source", default=os.environ.get("UPDATE_LOG_SOURCE", SOURCE))
    return parser.parse_args(argv)


def parse_kickoff() -> datetime:
    raw = os.environ.get("UPDATE_LOG_KICKOFF")
    if raw:
        try:
            value = datetime.fromisoformat(raw)
        except ValueError:
            pass
        else:
            return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.now(UTC)


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
    run_kickoff = parse_kickoff()
    result_file = os.environ.get("SCS_RESULT_FILE")

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

    db = None
    try:
        from scs_hub import SportsHub

        db = SportsHub(db_con=args.db_con).db
    except ImportError as exc:
        if result_file is None:
            print(
                "error: scripts.job requires scs-hub; run with "
                "`uv run --group data python -m scripts.job`",
                flush=True,
            )
            return 1
        print(
            f"warning: scs-hub unavailable: {exc}; row_count unavailable",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - status logging must not mask this
        if result_file is None:
            print(f"error: could not connect to [{args.db_con}]: {exc}", flush=True)
            return 1
        print(
            f"warning: could not connect to [{args.db_con}]: {exc}; row_count unavailable",
            flush=True,
        )

    row_count = None
    if db is not None and error is None:
        try:
            row_count = db.read("SELECT count(*) AS n FROM mod.player_archetype")[0, "n"]
        except Exception as exc:  # noqa: BLE001 - still write the run status
            print(f"warning: could not count mod.player_archetype: {exc}", flush=True)

    if result_file is not None:
        payload = {"row_count": row_count}
        if error is not None:
            payload["error"] = error
        try:
            Path(result_file).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write SCS_RESULT_FILE: {exc}", flush=True)
            return 1
    else:
        if db is None:
            return 1
        try:
            log_run(
                db,
                kickoff=run_kickoff,
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
        log(f"completed; mod.player_archetype has {row_count:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
