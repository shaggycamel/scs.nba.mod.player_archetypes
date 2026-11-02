"""Crosswalks between the `nba`, `statyx` and `util` id spaces.

The parquet dumps under `data/` come from three schemas that share no
identifiers:

    nba     game_id 21501032   player_id 200811
    statyx  game_id 900034932  player_id 1253150

`util.player_id_map_vw` bridges players. Games have no published map, so we
rebuild one from (game_date, home_abbr, away_abbr), which resolves 5243/5289
statyx games; the residue is play-in games the nba schedule labels elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl


def _default_data_dir() -> Path:
    """Locate the Parquet snapshot.

    `PLAYER_ARCHETYPES_DATA` wins when set. Otherwise the package expects a
    sibling `data/` directory in the checkout, which is where the original
    nba_adhoc snapshot lived.
    """
    configured = os.environ.get("PLAYER_ARCHETYPES_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "data"


DATA = _default_data_dir()


def player_map(data: Path = DATA) -> pl.DataFrame:
    """statyx_id <-> nba_id, for players carrying both."""
    return (
        pl.read_parquet(data / "util" / "player_id_map_vw.parquet")
        .filter(pl.col("statyx_id").is_not_null() & pl.col("nba_id").is_not_null())
        .select("player_key", "conformed_name", "nba_id", "statyx_id")
        .unique(subset=["statyx_id"])
    )


def nba_schedule(data: Path = DATA) -> pl.DataFrame:
    """One row per nba game, with date, season and home/away abbreviations.

    The raw table is team-level (two rows per game); we keep the home row so
    that `team` is the home side and `opponent` the visitor.
    """
    return (
        pl.read_parquet(data / "nba" / "league_game_schedule.parquet")
        .filter(pl.col("home"))
        .select(
            pl.col("game_id").alias("nba_game_id"),
            "game_date",
            "season",
            "season_type",
            pl.col("team").alias("home_abbr"),
            pl.col("opponent").alias("away_abbr"),
        )
        .unique(subset=["nba_game_id"])
    )


def game_map(data: Path = DATA) -> pl.DataFrame:
    """statyx game_id -> nba game_id, joined on date and the two abbreviations."""
    statyx = pl.read_parquet(data / "statyx" / "schedule.parquet").select(
        pl.col("game_id").alias("sx_game_id"),
        "game_date",
        pl.col("home_team_abbr").alias("home_abbr"),
        pl.col("visitor_team_abbr").alias("away_abbr"),
    )
    return (
        statyx.join(
            nba_schedule(data), on=["game_date", "home_abbr", "away_abbr"], how="inner"
        )
        .select("sx_game_id", "nba_game_id", "game_date", "season", "season_type")
        .unique(subset=["sx_game_id"])
    )
