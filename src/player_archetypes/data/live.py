"""Read the pipeline inputs from the live database.

The SQL is deliberately narrow: the production pipeline needs the box-score
columns that feed the rate features and the schedule columns that attach a
date, season and season type. Keeping the selection explicit means a new
column upstream cannot silently change the model inputs.
"""
from __future__ import annotations

import polars as pl

BOX_SCORE_SQL = """
    SELECT
        game_id,
        team_abbreviation,
        player_id,
        player_name,
        start_position,
        comment,
        min,
        fga,
        fg3_a,
        fta,
        ast,
        ast_pct,
        ast_ratio,
        tov_pct,
        oreb_pct,
        dreb_pct,
        stl,
        blk,
        pf,
        usg_pct,
        e_usg_pct
    FROM nba.player_box_score
    WHERE min IS NOT NULL AND min > 0
"""

SCHEDULE_SQL = """
    SELECT
        game_id,
        game_date,
        season,
        season_type,
        team,
        opponent,
        home
    FROM nba.league_game_schedule
"""


def read_box_scores(db) -> pl.DataFrame:
    """Played player-games with only the columns the archetype features use."""
    return db.read(BOX_SCORE_SQL)


def read_schedule(db) -> pl.DataFrame:
    """The team-level schedule, used to attach date, season and season type."""
    return db.read(SCHEDULE_SQL)
