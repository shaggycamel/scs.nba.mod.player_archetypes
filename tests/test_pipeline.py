from __future__ import annotations

from datetime import date

import polars as pl

from player_archetypes.data import load
from scripts.pipeline import regular_season_only


def test_regular_season_filter_is_applied():
    box = pl.DataFrame(
        {
            "season_type": ["Regular Season", "Pre Season", "Playoffs", "All Star"],
            "game_id": [1, 2, 3, 4],
        }
    )

    filtered = regular_season_only(box, True)

    assert filtered["game_id"].to_list() == [1]
    assert regular_season_only(box, False).height == box.height


def test_box_scores_from_frames_joins_schedule_and_drops_dnps():
    schedule = pl.DataFrame(
        {
            "game_id": [1, 1],
            "game_date": [date(2026, 10, 20), date(2026, 10, 20)],
            "season": ["2026-27", "2026-27"],
            "season_type": ["Regular Season", "Regular Season"],
            "team": ["HOM", "AWY"],
            "opponent": ["AWY", "HOM"],
            "home": [True, False],
        }
    )
    box = pl.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 11],
            "player_name": ["Played", "DNP"],
            "team_abbreviation": ["HOM", "HOM"],
            "start_position": ["G", None],
            "comment": [None, "DNP"],
            "min": [30.0, None],
            "fga": [10.0, None],
            "fg3_a": [2.0, None],
            "fta": [4.0, None],
            "ast": [3.0, None],
            "ast_pct": [0.2, None],
            "ast_ratio": [1.0, None],
            "tov_pct": [0.1, None],
            "oreb_pct": [0.05, None],
            "dreb_pct": [0.1, None],
            "stl": [1.0, None],
            "blk": [0.0, None],
            "pf": [2.0, None],
            "usg_pct": [0.0, 0.0],
            "e_usg_pct": [0.0, 0.0],
        }
    )

    out = load.box_scores_from_frames(box, schedule)

    assert out.height == 1
    assert out["player_id"].to_list() == [10]
    assert out["game_date"].to_list() == [date(2026, 10, 20)]
    assert out["season_type"].to_list() == ["Regular Season"]
