from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from player_archetypes import features


def per_game_frame(games: int = 8) -> pl.DataFrame:
    start = date(2024, 1, 1)
    return pl.DataFrame(
        [
            {
                "game_id": game,
                "player_id": 1,
                "player_name": "Test Player",
                "game_date": start + timedelta(days=game),
                "season": "2024-25",
                "team_abbreviation": "TST",
                "min": 30.0,
                "fg3_a": 2.0,
                "fga": 10.0,
                "fta": 4.0,
                "ast": 5.0,
                "stl": 1.0,
                "blk": 0.0,
                "pf": 2.0,
                "start_position": "G",
                "usg_pct": 0.20,
                "ast_pct": 0.20,
                "ast_ratio": 2.0,
                "tov_pct": 0.10,
                "oreb_pct": 0.02,
                "dreb_pct": 0.10,
            }
            for game in range(games)
        ]
    )


def test_rolling_features_never_see_the_current_game():
    raw = per_game_frame()
    base = features.per_game(raw)
    before = features.rolling(base, window=4, min_periods=2)

    changed = features.per_game(
        raw.with_columns(
            pl.when(pl.col("game_id") == 5)
            .then(pl.col("fga") + 100.0)
            .otherwise(pl.col("fga"))
            .alias("fga")
        )
    )
    after = features.rolling(changed, window=4, min_periods=2)

    at_game = lambda frame, game: frame.filter(pl.col("game_id") == game)["r_fga_per_min"].item()
    assert at_game(before, 5) == pytest.approx(at_game(after, 5))
    assert at_game(before, 6) != pytest.approx(at_game(after, 6))


def test_usage_history_never_sees_the_current_game():
    base = features.per_game(per_game_frame())
    base = base.with_columns(pl.lit(0.20).alias("usg_pct"))
    before = features.usage_history(base)

    changed = base.with_columns(
        pl.when(pl.col("game_id") == 5)
        .then(pl.lit(0.50))
        .otherwise(pl.col("usg_pct"))
        .alias("usg_pct")
    )
    after = features.usage_history(changed)

    at_game = lambda frame, game: frame.filter(pl.col("game_id") == game)["usg_last5"].item()
    assert at_game(before, 5) == pytest.approx(at_game(after, 5))
    assert at_game(before, 6) != pytest.approx(at_game(after, 6))


def test_feature_names_exclude_usage_and_optional_blocks():
    names = features.feature_names()
    assert not any("usg" in name for name in names)
    assert "r_min" in names
    assert "r_min" not in features.feature_names(include_volume=False)
    assert "r_fg_pct" not in names
    assert "r_fg_pct" in features.feature_names(include_efficiency=True)
