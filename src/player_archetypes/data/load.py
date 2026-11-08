"""Cleaned loaders. Each one fixes a defect found in the raw dumps.

Defects handled here (see scripts/audit.py for the evidence):

1. `nba.player_box_score` encodes DNPs as `usg_pct = 0.0` with a null `min`,
   not as missing rows. 105394 / 589082 rows (17.9%) are DNPs, so a naive mean
   usage reads 0.151 against a played-only 0.1839. `box_scores` flags them.
2. `statyx.advanced_stats` is duplicated ~2x by a bad season join: 116972 rows
   collapse to 57255 unique (player_id, game_id), and both season labels span
   the identical date range. The `season` column is discarded and rebuilt from
   the game map; conflicting duplicates are averaged.
3. statyx ids are translated into the nba id space via `player_archetypes.data.ids`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from player_archetypes.data.ids import (
    DATA,
    game_map,
    nba_schedule_from_frame,
    player_map,
)

# usg_pct / e_usg_pct are the modelling target; poss and pace are team-level
# tempo, not individual archetype. None of these may enter the clustering inputs.
TARGET_COLS = ("usg_pct", "e_usg_pct")


def box_scores_from_frames(
    box: pl.DataFrame,
    schedule: pl.DataFrame,
    played_only: bool = True,
) -> pl.DataFrame:
    """Player-game box scores joined to schedule metadata and DNP handling.

    `played` is False where the player was on the roster but did not play. Those
    rows carry a real `comment` (the DNP reason) and a fabricated `usg_pct` of
    0.0, so they are dropped by default.
    """
    sched = nba_schedule_from_frame(schedule).select(
        pl.col("nba_game_id").alias("game_id"), "game_date", "season", "season_type"
    )
    out = (
        box.join(sched, on="game_id", how="left")
        .with_columns(
            played=pl.col("min").is_not_null() & (pl.col("min") > 0),
            dnp_reason=pl.when(pl.col("comment").is_null() | (pl.col("comment") == ""))
            .then(None)
            .otherwise(pl.col("comment")),
        )
        # DNP rows claim 0.0 usage; blank it so nothing averages over a fiction.
        .with_columns(
            [
                pl.when(pl.col("played")).then(pl.col(c)).otherwise(None).alias(c)
                for c in TARGET_COLS
            ]
        )
    )
    return out.filter(pl.col("played")) if played_only else out


def box_scores(data: Path = DATA, played_only: bool = True) -> pl.DataFrame:
    """Player-game box scores from the Parquet snapshot."""
    return box_scores_from_frames(
        pl.read_parquet(data / "nba" / "player_box_score.parquet"),
        pl.read_parquet(data / "nba" / "league_game_schedule.parquet"),
        played_only=played_only,
    )


def advanced_stats(data: Path = DATA) -> pl.DataFrame:
    """Per-game tracking stats, de-duplicated and mapped to nba ids.

    Returns one row per (nba player_id, nba game_id). `period` is always 0 in
    the source, i.e. these are whole-game rows despite the column name.
    """
    raw = pl.read_parquet(data / "statyx" / "advanced_stats.parquet")
    key = ["player_id", "game_id"]
    numeric = [c for c, dt in raw.schema.items() if dt.is_numeric() and c not in (*key, "period")]
    deduped = raw.group_by(key).agg([pl.col(c).mean().alias(c) for c in numeric])

    pmap = player_map(data).select("statyx_id", "nba_id")
    gmap = game_map(data).select("sx_game_id", "nba_game_id", "game_date", "season")
    return (
        deduped.join(pmap, left_on="player_id", right_on="statyx_id", how="inner")
        .join(gmap, left_on="game_id", right_on="sx_game_id", how="inner")
        .drop("player_id", "game_id", "statyx_id", "sx_game_id")
        .rename({"nba_id": "player_id", "nba_game_id": "game_id"})
    )


def play_types(data: Path = DATA) -> pl.DataFrame:
    """Season-level play-type mix, mapped to nba ids and converted to shares.

    Only 2025-26 exists, and the grain is the whole season, so these features
    leak within a season. Use them for interpretation, or lagged by a full
    season -- never as same-season model inputs.
    """
    raw = pl.read_parquet(data / "statyx" / "play_types.parquet")
    kinds = [
        "spot_up",
        "iso",
        "transition",
        "p_and_r_ball_handler",
        "p_and_r_roll_man",
        "cut",
        "hand_off",
        "off_screen",
        "post_up",
        "o_board",
    ]
    total = pl.sum_horizontal([pl.col(f"{k}_attempts") for k in kinds])
    pmap = player_map(data).select("statyx_id", "nba_id")
    return (
        raw.with_columns(pt_total_attempts=total)
        .with_columns(
            [
                (pl.col(f"{k}_attempts") / pl.col("pt_total_attempts")).alias(f"pt_share_{k}")
                for k in kinds
            ]
        )
        .join(pmap, left_on="player_id", right_on="statyx_id", how="inner")
        .select(
            pl.col("nba_id").alias("player_id"),
            "season",
            "player_name",
            "pt_total_attempts",
            *[f"pt_share_{k}" for k in kinds],
        )
    )


def listed_positions(data: Path = DATA) -> pl.DataFrame:
    """Listed position per player-season, the baseline the clusters replace."""
    return (
        pl.read_parquet(data / "nba" / "player_info.parquet")
        .select("season", "player_id", "position", "height_cm", "weight_kg", "season_exp")
        .filter(pl.col("position").is_not_null() & (pl.col("position") != ""))
    )
