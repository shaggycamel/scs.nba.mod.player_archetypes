from __future__ import annotations

import polars as pl

from player_archetypes.naming import describe_archetypes, name_archetype


def test_name_archetype_uses_the_signature_thresholds():
    assert (
        name_archetype(
            {
                "r_oreb_pct": 1.2,
                "r_fg3a_share": -1.2,
            }
        )
        == "Rim-running center"
    )
    assert (
        name_archetype(
            {
                "r_ast_pct": 1.2,
                "r_min": 0.9,
                "r_fga_per_min": 0.6,
            }
        )
        == "Primary creator"
    )
    assert name_archetype({}) == "Connective wing"


def test_describe_archetypes_inserts_description_after_label_and_dedupes():
    profiles = pl.DataFrame(
        {
            "archetype_label": [0, 1],
            "n": [10, 20],
            "r_start_rate": [-1.0, -1.0],
            "r_fg3a_share": [0.0, 0.0],
            "r_oreb_pct": [0.0, 0.0],
            "r_blk_per_min": [0.0, 0.0],
            "r_ast_pct": [0.0, 0.0],
            "r_ast_ratio": [0.0, 0.0],
            "r_min": [0.0, 0.0],
            "r_fga_per_min": [0.0, 0.0],
        }
    )

    described = describe_archetypes(profiles)

    assert described.columns[:3] == [
        "archetype_label",
        "archetype_description",
        "n",
    ]
    assert described["archetype_description"].to_list() == [
        "Bench wing",
        "Bench wing 2",
    ]
