"""Refresh the Parquet snapshot from the live database.

This is the data stage of the scheduled job. It uses the read-only
``cockroach-read`` connection from scs-hub and pulls only the tables the
archetype model needs, rather than every object in the database.

Usage:
    export SCS_HUB_CREDENTIALS=~/.config/scs_hub_credentials.ini
    uv run --group data python -m scripts.refresh_data

The output directory is ``PLAYER_ARCHETYPES_DATA`` when set, otherwise
``data/`` in the repository root. A ``_snapshot.json`` manifest is written
after every table has landed, so a partial refresh never looks complete.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Only the objects the loaders read. Keeping this list explicit means the
# refresh is bounded and a new table in the database cannot silently change
# what the model sees.
TABLES: tuple[tuple[str, str], ...] = (
    ("nba", "player_box_score"),
    ("nba", "league_game_schedule"),
    ("nba", "player_info"),
    ("statyx", "advanced_stats"),
    ("statyx", "play_types"),
    ("statyx", "schedule"),
    ("util", "player_id_map_vw"),
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _sports_hub():
    """Import lazily so the rest of the module works without the data group.

    scs-hub imports ``nbainjuries`` eagerly, and that needs a JVM. Keeping the
    import inside this function means the refresh script can still be imported
    (and tested) when only the core model dependencies are installed.
    """
    from scs_hub import SportsHub

    return SportsHub


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_version() -> str | None:
    try:
        return version("player-archetypes")
    except PackageNotFoundError:
        return None


def _default_out() -> Path:
    configured = os.environ.get("PLAYER_ARCHETYPES_DATA")
    return Path(configured).expanduser() if configured else ROOT / "data"


def _max_date(frame: pl.DataFrame | None, played_only: bool = False) -> str | None:
    if frame is None or not frame.height or "game_date" not in frame.columns:
        return None
    if played_only and "team_winner" in frame.columns:
        frame = frame.filter(pl.col("team_winner").is_not_null())
    if not frame.height:
        return None
    value = frame["game_date"].max()
    return str(value) if value is not None else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_out(),
        help="snapshot root; defaults to PLAYER_ARCHETYPES_DATA or ./data",
    )
    parser.add_argument(
        "--db-con",
        default="cockroach-read",
        help="credentials.ini section to connect with",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    tmp_dir = out_dir / ".refresh-tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    SportsHub = _sports_hub()
    log(f"connecting with [{args.db_con}]")
    hub = SportsHub(db_con=args.db_con)

    manifest_tables: dict[str, dict[str, object]] = {}
    schedule: pl.DataFrame | None = None
    try:
        for schema, table in TABLES:
            if not IDENTIFIER.match(schema) or not IDENTIFIER.match(table):
                raise ValueError(f"unsafe identifier: {schema}.{table}")
            log(f"reading {schema}.{table}")
            frame = hub.db.read(f"SELECT * FROM {schema}.{table}")
            target = tmp_dir / schema / f"{table}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(target)
            if schema == "nba" and table == "league_game_schedule":
                schedule = frame
            manifest_tables[f"{schema}.{table}"] = {
                "rows": frame.height,
                "columns": frame.width,
                "path": f"{schema}/{table}.parquet",
            }
            log(f"  {frame.height:,} rows x {frame.width} columns")
    except Exception as exc:  # noqa: BLE001 - clean up staging before re-raising
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"error: refresh failed: {exc}", file=sys.stderr, flush=True)
        return 1

    # Every table is in the staging directory. Move them into the snapshot
    # together, then write the manifest last.
    for schema, table in TABLES:
        source = tmp_dir / schema / f"{table}.parquet"
        destination = out_dir / schema / f"{table}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    max_game_date = _max_date(schedule, played_only=True)
    max_scheduled_game_date = _max_date(schedule)

    manifest = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "db_con": args.db_con,
        "snapshot_dir": str(out_dir),
        "max_game_date": max_game_date,
        "max_scheduled_game_date": max_scheduled_game_date,
        "git_sha": _git_sha(),
        "package_version": _package_version(),
        "tables": manifest_tables,
    }
    manifest_path = out_dir / "_snapshot.json"
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp_manifest.replace(manifest_path)

    log(
        f"refreshed {len(TABLES)} tables; last played {max_game_date or 'unknown'}, "
        f"schedule through {max_scheduled_game_date or 'unknown'} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
