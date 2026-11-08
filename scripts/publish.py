"""Publish the pipeline artifacts to CockroachDB through scs-hub.

Usage:
    uv run --group data python -m scripts.publish
    uv run --group data python -m scripts.publish --all-seasons  # retrospective

The output lives in the ``mod`` schema. By default only the current season is
published, because assigning old seasons with a model fitted on later data is a
retrospective backcast, not a point-in-time result. ``--all-seasons`` writes
that backcast deliberately.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

import polars as pl
from sqlalchemy import text

from player_archetypes.naming import describe_archetypes

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
KEY_COLUMNS = (
    "game_id",
    "player_id",
    "player_name",
    "game_date",
    "season",
    "team_abbreviation",
    "prior_games",
)
SCALAR_ARCHETYPE_COLUMNS = ("archetype_label", "archetype_confidence", "archetype_entropy")
PUBLIC_TABLES = (
    "player_archetype",
    "player_archetype_current",
    "player_archetype_profile",
    "player_archetype_model",
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def identifier(value: str, kind: str) -> str:
    if not IDENTIFIER.match(value):
        raise ValueError(f"unsafe {kind}: {value!r}")
    return value


def table_name(schema: str, table: str) -> str:
    return f"{identifier(schema, 'schema')}.{identifier(table, 'table')}"


def base_table(schema: str, table: str) -> str:
    """The writable table behind its public view."""
    return table_name(schema, f"{table}_base")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/pipeline"),
        help="directory containing assignments.parquet, archetype_profiles.parquet and run.json",
    )
    parser.add_argument("--db-con", default="cockroach-write")
    parser.add_argument("--schema", default="mod")
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="write every historical season as a retrospective backcast",
    )
    parser.add_argument(
        "--no-grant",
        action="store_true",
        help="skip SELECT grants for the read-only role",
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
    archetype_table = base_table(schema, "player_archetype")
    current_table = base_table(schema, "player_archetype_current")
    profile_table = base_table(schema, "player_archetype_profile")
    model_table = base_table(schema, "player_archetype_model")

    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {archetype_table} (
            game_id INT8 NOT NULL,
            player_id INT8 NOT NULL,
            player_name STRING,
            game_date DATE NOT NULL,
            season STRING,
            team_abbreviation STRING,
            prior_games INT8,
            archetype_label INT8,
            archetype_confidence FLOAT8,
            archetype_entropy FLOAT8,
        {archetype_defs},
        {feature_defs},
            model_cutoff DATE,
            PRIMARY KEY (game_id, player_id)
        )
        """
    )
    db.execute(
        f"CREATE INDEX IF NOT EXISTS player_archetype_player_date_idx "
        f"ON {archetype_table} (player_id, game_date)"
    )
    db.execute(
        f"CREATE INDEX IF NOT EXISTS player_archetype_season_idx ON {archetype_table} (season)"
    )
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {current_table} (
            player_id INT8 NOT NULL PRIMARY KEY,
            player_name STRING,
            game_id INT8,
            game_date DATE,
            season STRING,
            team_abbreviation STRING,
            prior_games INT8,
            games_season INT8,
            archetype_label INT8,
            archetype_confidence FLOAT8,
            archetype_entropy FLOAT8,
        {archetype_defs},
        {feature_defs},
            model_cutoff DATE
        )
        """
    )
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {profile_table} (
            archetype_label INT8 NOT NULL PRIMARY KEY,
            archetype_description STRING,
            n INT8,
        {feature_defs},
            model_cutoff DATE
        )
        """
    )
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {model_table} (
            model_cutoff DATE NOT NULL PRIMARY KEY,
            generated_at TIMESTAMPTZ,
            git_sha STRING,
            k INT8,
            rolling_window INT8,
            min_periods INT8,
            min_prior_games INT8,
            season_type STRING,
            seed INT8,
            fit_rows INT8,
            assignments_rows INT8,
            mean_confidence FLOAT8,
            frac_ambiguous FLOAT8
        )
        """
    )


def create_views(db, schema: str) -> None:
    """Expose the public names as read-only views over the _base tables."""
    for table in PUBLIC_TABLES:
        db.execute(
            f"CREATE OR REPLACE VIEW {table_name(schema, table)} AS "
            f"SELECT * FROM {base_table(schema, table)}"
        )


def prepare_assignments(frame: pl.DataFrame, model_cutoff: date) -> pl.DataFrame:
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
            pl.lit(model_cutoff).alias("model_cutoff"),
        )
        .sort(["game_date", "game_id", "player_id"])
    )


def prepare_profiles(frame: pl.DataFrame, model_cutoff: date) -> pl.DataFrame:
    if "archetype_description" not in frame.columns:
        frame = describe_archetypes(frame)
    features = sorted(column for column in frame.columns if column.startswith("r_"))
    return (
        frame.with_columns(
            pl.col("archetype_label").cast(pl.Int64),
            pl.col("n").cast(pl.Int64),
            pl.lit(model_cutoff).alias("model_cutoff"),
        )
        .select(
            [
                "archetype_label",
                "archetype_description",
                "n",
                *features,
                "model_cutoff",
            ]
        )
        .sort("archetype_label")
    )


def prepare_current(assignments: pl.DataFrame, season: str) -> pl.DataFrame:
    latest = (
        assignments.filter(pl.col("season") == season)
        .sort(["player_id", "game_date", "game_id"])
        .group_by("player_id")
        .last()
    )
    games = (
        assignments.filter(pl.col("season") == season)
        .group_by("player_id")
        .agg(pl.len().alias("games_season"))
    )
    return latest.join(games, on="player_id").sort("player_id")


def prepare_model(meta: dict, model_cutoff: date) -> pl.DataFrame:
    model = meta["model"]
    assignments = meta["assignments"]
    row = {
        "model_cutoff": model_cutoff,
        "generated_at": datetime.fromisoformat(meta["generated_at"]),
        "git_sha": meta.get("git_sha"),
        "k": model.get("k"),
        "rolling_window": model.get("window"),
        "min_periods": model.get("min_periods"),
        "min_prior_games": model.get("min_prior_games"),
        "season_type": model.get("season_type"),
        "seed": model.get("seed"),
        "fit_rows": model.get("fit_rows"),
        "assignments_rows": assignments.get("rows"),
        "mean_confidence": assignments.get("mean_confidence"),
        "frac_ambiguous": assignments.get("frac_ambiguous"),
    }
    return pl.DataFrame([row], schema_overrides={"git_sha": pl.String})


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
    """Delete by season so CockroachDB stays inside its transaction lock budget."""
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
    profiles: pl.DataFrame,
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

    model_cutoff = date.fromisoformat(meta["model"]["cutoff"])
    assignments = prepare_assignments(assignments, model_cutoff)
    profiles = prepare_profiles(profiles, model_cutoff)
    current_season = assignments["season"].max()
    if current_season is None:
        raise ValueError("assignments contain no season")

    validate_assignments(assignments, archetypes)
    create_schema(db, schema)
    create_tables(db, schema, archetypes, features)
    create_views(db, schema)

    archetype_base = base_table(schema, "player_archetype")
    archetype_public = table_name(schema, "player_archetype")
    current = assignments.filter(pl.col("season") == current_season)
    if all_seasons:
        log(f"writing retrospective backcast: {assignments.height:,} assignment rows")
        delete_all_batched(db, schema, "player_archetype_base")
        write_seasons(db, schema, assignments)
    else:
        old_seasons = (
            db.read(f"SELECT DISTINCT season FROM {archetype_base}").get_column("season").to_list()
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

    current_archetypes = prepare_current(assignments, current_season)
    log(f"writing player_archetype_current: {current_archetypes.height:,} rows")
    replace_table_atomic(db, schema, "player_archetype_current_base", current_archetypes)

    log(f"writing player_archetype_profile: {profiles.height:,} rows")
    replace_table_atomic(db, schema, "player_archetype_profile_base", profiles)

    model = prepare_model(meta, model_cutoff)
    log("writing player_archetype_model")
    replace_table_atomic(
        db,
        schema,
        "player_archetype_model_base",
        model,
        where="model_cutoff = :cutoff",
        params={"cutoff": model_cutoff},
    )

    actual = db.read(
        f"SELECT count(*) AS n FROM {archetype_base} WHERE season = '{current_season}'"
    )[0, "n"]
    expected = current.height if not all_seasons else assignments.height
    if actual != expected:
        raise ValueError(f"published row count {actual} != expected {expected}")

    written = assignments if all_seasons else current
    log(
        f"published {written.height:,} rows across "
        f"{written['season'].n_unique()} season(s) to {archetype_public}; "
        f"current season {current_season}"
    )


def grant_read(db, schema: str) -> None:
    schema = identifier(schema, "schema")
    db.execute(f"GRANT USAGE ON SCHEMA {schema} TO kobe_public")
    for table in PUBLIC_TABLES:
        db.execute(f"GRANT SELECT ON TABLE {table_name(schema, table)} TO kobe_public")
        db.execute(f"REVOKE SELECT ON TABLE {base_table(schema, table)} FROM kobe_public")
    log("granted SELECT on mod views to kobe_public")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser()
    assignments = pl.read_parquet(out / "assignments.parquet")
    profiles = pl.read_parquet(out / "archetype_profiles.parquet")
    meta = json.loads((out / "run.json").read_text())

    from scs_hub import SportsHub

    hub = SportsHub(db_con=args.db_con)
    publish(hub.db, args.schema, assignments, profiles, meta, args.all_seasons)
    if not args.no_grant:
        grant_read(hub.db, args.schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
