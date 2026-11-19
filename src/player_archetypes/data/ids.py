"""Crosswalks between the `nba`, `statyx` and `util` id spaces.

The three schemas share no identifiers:

    nba     game_id 21501032   player_id 200811
    statyx  game_id 900034932  player_id 1253150

`util.player_id_map_vw` bridges players. Games have no published map, so we
rebuild one from (game_date, home_abbr, away_abbr). The database is the only
source; there is no Parquet snapshot.
"""

from __future__ import annotations

import os

import polars as pl

DEFAULT_DB_CON = "jetson-read"

_DB = None


def db(db_con: str | None = None):
    """The shared read connection, created on first use.

    `scs_hub` is imported lazily so the core model and its tests do not need the
    data dependency installed.
    """
    global _DB
    if _DB is None:
        from scs_hub import SportsHub

        section = db_con or os.environ.get("PLAYER_ARCHETYPES_DB_CON", DEFAULT_DB_CON)
        _DB = SportsHub(db_con=section).db
    return _DB


def player_map(db_con: str | None = None) -> pl.DataFrame:
    """statyx_id <-> nba_id, for players carrying both."""
    frame = db(db_con).read(
        """
        SELECT player_key, conformed_name, nba_id, statyx_id
        FROM util.player_id_map_vw
        WHERE statyx_id IS NOT NULL AND nba_id IS NOT NULL
        """
    )
    return frame.select(
        "player_key",
        "conformed_name",
        pl.col("nba_id").cast(pl.Int64),
        pl.col("statyx_id").cast(pl.Int64),
    ).unique(subset=["statyx_id"])


def nba_schedule_from_frame(schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per nba game, with date, season and home/away abbreviations.

    The view is team-level (two rows per game); we keep the home row so that
    `team` is the home side and `opponent` the visitor.
    """
    return (
        schedule.filter(pl.col("home"))
        .select(
            pl.col("game_id").cast(pl.Int64).alias("nba_game_id"),
            "game_date",
            "season",
            "season_type",
            pl.col("team").alias("home_abbr"),
            pl.col("opponent").alias("away_abbr"),
        )
        .unique(subset=["nba_game_id"])
    )


def nba_schedule(db_con: str | None = None) -> pl.DataFrame:
    """One row per nba game, read from the public schedule view."""
    return nba_schedule_from_frame(
        db(db_con).read(
            """
            SELECT game_id, game_date, season, season_type, team, opponent, home
            FROM nba.league_game_schedule_vw
            """
        )
    )


def game_map(db_con: str | None = None) -> pl.DataFrame:
    """statyx game_id -> nba game_id, joined on date and the two abbreviations."""
    statyx = db(db_con).read(
        """
        SELECT
            game_id AS sx_game_id,
            game_date,
            home_team_abbr AS home_abbr,
            visitor_team_abbr AS away_abbr
        FROM statyx.schedule_vw
        """
    )
    return (
        statyx.join(nba_schedule(db_con), on=["game_date", "home_abbr", "away_abbr"], how="inner")
        .select("sx_game_id", "nba_game_id", "game_date", "season", "season_type")
        .unique(subset=["sx_game_id"])
    )
