from __future__ import annotations

from datetime import date

import polars as pl

from scripts.refresh_data import TABLES, _max_date


def test_tables_are_the_model_inputs():
    assert set(TABLES) == {
        ("nba", "player_box_score"),
        ("nba", "league_game_schedule"),
        ("nba", "player_info"),
        ("statyx", "advanced_stats"),
        ("statyx", "play_types"),
        ("statyx", "schedule"),
        ("util", "player_id_map_vw"),
    }


def test_max_date_can_ignore_unplayed_games():
    schedule = pl.DataFrame(
        {
            "game_date": [date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 2)],
            "team_winner": ["A", "B", None],
        }
    )

    assert _max_date(schedule) == "2026-11-02"
    assert _max_date(schedule, played_only=True) == "2026-10-31"


def test_max_date_handles_missing_data():
    assert _max_date(None) is None
    assert _max_date(pl.DataFrame({"game_id": [1]})) is None
