"""Publish archetype assignments to the analytics database through scs-hub.

Usage:
    uv run --group data python -m scripts.publish
    uv run --group data python -m scripts.publish --all-seasons  # retrospective

The output is one private table, ``mod.player_archetype_base``, and one public
view applied from ``nba/database/sql/views/mod``. By default only the current
season is written, because assigning old seasons with a model fitted on later
data is a retrospective backcast, not a point-in-time result.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import polars as pl
from sqlalchemy import text

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
KEY_COLUMNS = (
    "game_id",
    "player_id",
    "player_name",
    "game_date",
    "season",
    "team_abbreviation",
    "prior_games",
    "features_through",
    "model_version",
    "model_cutoff",
    "model_checksum",
)
SCALAR_ARCHETYPE_COLUMNS = ("archetype_label", "archetype_confidence", "archetype_entropy")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def identifier(value: str, kind: str) -> str:
    if not IDENTIFIER.match(value):
        raise ValueError(f"unsafe {kind}: {value!r}")
    return value


def table_name(schema: str, table: str) -> str:
    return f"{identifier(schema, 'schema')}.{identifier(table, 'table')}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/pipeline"),
        help="directory containing assignments.parquet and run.json",
    )
    parser.add_argument("--db-con", default="jetson-write")
    parser.add_argument("--schema", default="mod")
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="write every historical season as a retrospective backcast",
    )
    return parser.parse_args(argv)


def archetype_columns(frame: pl.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column.startswith("archetype_") and column.removeprefix("archetype_").isdigit()
    ]
    return sorted(columns, key=lambda column: int(column.removeprefix("archetype_")))


def feature_columns(frame: pl.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("r_"))


def create_schema(db, schema: str) -> None:
    db.execute(f"CREATE SCHEMA IF NOT EXISTS {identifier(schema, 'schema')}")


def ensure_private(db, target: str) -> None:
    """kobe_public reads the view; the base table must stay private."""
    has_role = db.read("SELECT count(*) AS n FROM pg_roles WHERE rolname = 'kobe_public'")[0, "n"]
    if not has_role:
        return
    db.execute(f"REVOKE ALL ON {target} FROM kobe_public")
    granted = db.read(f"SELECT has_table_privilege('kobe_public', '{target}', 'SELECT') AS n")[
        0, "n"
    ]
    if granted:
        raise RuntimeError(f"{target} is still readable by kobe_public")


def create_tables(
    db,
    schema: str,
    archetypes: list[str],
    features: list[str],
) -> None:
    schema = identifier(schema, "schema")
    archetype_defs = ",\n".join(
        f"    {identifier(c, 'archetype column')} FLOAT8" for c in archetypes
    )
    feature_defs = ",\n".join(f"    {identifier(c, 'feature column')} FLOAT8" for c in features)
    target = table_name(schema, "player_archetype_base")

    # TEXT rather than STRING: the analytics database is PostgreSQL, and STRING
    # is not a Postgres type.
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {target} (
            game_id INT8 NOT NULL,
            player_id INT8 NOT NULL,
            player_name TEXT,
            game_date DATE NOT NULL,
            season TEXT,
            team_abbreviation TEXT,
            prior_games INT8,
            features_through DATE,
            model_version TEXT NOT NULL,
            model_cutoff DATE,
            model_checksum TEXT,
            archetype_label INT8,
            archetype_confidence FLOAT8,
            archetype_entropy FLOAT8,
        {archetype_defs},
        {feature_defs},
            PRIMARY KEY (game_id, player_id)
        )
        """
    )
    db.execute(
        f"CREATE INDEX IF NOT EXISTS player_archetype_player_date_idx "
        f"ON {target} (player_id, game_date)"
    )
    db.execute(f"CREATE INDEX IF NOT EXISTS player_archetype_season_idx ON {target} (season)")
    ensure_private(db, target)


def prepare_assignments(frame: pl.DataFrame) -> pl.DataFrame:
    required = [*KEY_COLUMNS, *SCALAR_ARCHETYPE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"assignments missing columns: {missing}")

    return (
        frame.filter(
            pl.col("game_id").is_not_null()
            & pl.col("player_id").is_not_null()
            & pl.col("game_date").is_not_null()
        )
        .with_columns(
            pl.col("prior_games").cast(pl.Int64),
            pl.col("archetype_label").cast(pl.Int64),
            pl.col("model_cutoff").cast(pl.Date),
        )
        .sort(["game_date", "game_id", "player_id"])
    )


def validate_assignments(frame: pl.DataFrame, archetypes: list[str]) -> None:
    if frame.height == 0:
        raise ValueError("assignments are empty")
    keys = frame.select("game_id", "player_id")
    if keys.null_count().sum_horizontal()[0]:
        raise ValueError("assignments contain null game_id or player_id")
    duplicates = (
        frame.group_by("game_id", "player_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    )
    if duplicates.height:
        raise ValueError(f"assignments contain {duplicates.height} duplicate keys")
    if frame["features_through"].null_count():
        raise ValueError("assignments contain null features_through")
    if frame["model_version"].null_count() or not frame["model_version"].str.len_chars().min():
        raise ValueError("assignments contain an empty model_version")
    sums = frame.select(pl.sum_horizontal(archetypes).alias("total"))["total"]
    if ((sums - 1.0).abs() > 1e-6).any():
        raise ValueError("archetype responsibilities do not sum to 1")


def replace_table_atomic(
    db,
    schema: str,
    table: str,
    frame: pl.DataFrame,
    where: str = "true",
    params: dict | None = None,
) -> None:
    """Delete and insert in one transaction, so readers never see an empty table."""
    target = table_name(schema, table)
    with db.engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {target} WHERE {where}"), params or {})
        if frame.height:
            frame.write_database(target, conn, if_table_exists="append")


def write_seasons(db, schema: str, frame: pl.DataFrame) -> None:
    seasons = sorted(season for season in frame["season"].unique().to_list() if season)
    for season in seasons:
        chunk = frame.filter(pl.col("season") == season)
        log(f"  writing {season}: {chunk.height:,} rows")
        db.write(chunk, "player_archetype_base", schema=schema)


def delete_all_batched(db, schema: str, table: str) -> None:
    """Delete by season so the transaction stays inside the lock budget."""
    seasons = (
        db.read(f"SELECT DISTINCT season FROM {table_name(schema, table)}")
        .get_column("season")
        .to_list()
    )
    for season in seasons:
        where = "season IS NULL" if season is None else f"season = '{season}'"
        log(f"  clearing {season or 'null season'}")
        db.delete_where(table, schema, where)


def publish(
    db,
    schema: str,
    assignments: pl.DataFrame,
    meta: dict,
    all_seasons: bool,
) -> None:
    schema = identifier(schema, "schema")
    archetypes = archetype_columns(assignments)
    features = feature_columns(assignments)
    if not archetypes:
        raise ValueError("assignments contain no archetype_<n> columns")
    if not features:
        raise ValueError("assignments contain no rolling feature columns")

    assignments = prepare_assignments(assignments)
    current_season = assignments["season"].max()
    if current_season is None:
        raise ValueError("assignments contain no season")

    validate_assignments(assignments, archetypes)
    create_schema(db, schema)
    create_tables(db, schema, archetypes, features)

    target = table_name(schema, "player_archetype_base")
    current = assignments.filter(pl.col("season") == current_season)
    if all_seasons:
        log(f"writing retrospective backcast: {assignments.height:,} rows")
        delete_all_batched(db, schema, "player_archetype_base")
        write_seasons(db, schema, assignments)
    else:
        old_seasons = (
            db.read(f"SELECT DISTINCT season FROM {target}").get_column("season").to_list()
        )
        for season in old_seasons:
            if season != current_season:
                where = "season IS NULL" if season is None else f"season = '{season}'"
                log(f"  clearing historical {season or 'null season'}")
                db.delete_where("player_archetype_base", schema, where)
        log(f"atomically replacing {current_season}: {current.height:,} rows")
        replace_table_atomic(
            db,
            schema,
            "player_archetype_base",
            current,
            where="season = :season",
            params={"season": current_season},
        )

    actual = db.read(f"SELECT count(*) AS n FROM {target} WHERE season = '{current_season}'")[
        0, "n"
    ]
    expected = current.height if not all_seasons else assignments.height
    if actual != expected:
        raise ValueError(f"published row count {actual} != expected {expected}")

    written = assignments if all_seasons else current
    log(
        f"published {written.height:,} rows across "
        f"{written['season'].n_unique()} season(s) to {target}; "
        f"model {meta['model']['version']}; current season {current_season}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser()
    assignments = pl.read_parquet(out / "assignments.parquet")
    meta = json.loads((out / "run.json").read_text())

    from scs_hub import SportsHub

    hub = SportsHub(db_con=args.db_con)
    publish(hub.db, args.schema, assignments, meta, args.all_seasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
