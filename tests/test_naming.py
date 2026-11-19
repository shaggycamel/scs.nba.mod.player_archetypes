from __future__ import annotations

import polars as pl

from player_archetypes.naming import describe_archetypes, name_archetype


def test_name_archetype_uses_style_signatures():
    assert (
        name_archetype(
            {
                "r_oreb_pct": 1.4,
                "r_blk_per_min": 1.0,
                "r_fg3a_share": -1.4,
            }
        )
        == "Rim-running center"
    )
    assert (
        name_archetype(
            {
                "r_oreb_pct": 1.1,
                "r_blk_per_min": 1.3,
                "r_fg3a_share": -0.7,
            }
        )
        == "Rim-protecting big"
    )
    assert (
        name_archetype(
            {
                "r_oreb_pct": 0.5,
                "r_blk_per_min": 0.6,
                "r_ast_pct": -1.0,
            }
        )
        == "Interior finisher"
    )
    assert name_archetype({"r_ast_pct": 2.0, "r_ast_ratio": 1.2}) == "Playmaking guard"
    assert (
        name_archetype(
            {
                "r_ast_pct": 0.8,
                "r_ast_ratio": 1.3,
                "r_fg3a_share": 0.4,
            }
        )
        == "Secondary creator"
    )
    assert name_archetype({"r_ast_pct": 0.9, "r_fga_per_min": 1.1}) == "Scoring wing"
    assert (
        name_archetype(
            {
                "r_stl_per_min": 0.8,
                "r_ftr": -0.7,
                "r_blk_per_min": -0.5,
            }
        )
        == "Perimeter defensive wing"
    )
    assert name_archetype({"r_fg3a_share": 0.7, "r_ast_pct": -0.6}) == "Floor spacer"
    assert (
        name_archetype(
            {
                "r_dreb_pct": 0.8,
                "r_fga_per_min": 0.7,
                "r_fg3a_share": -0.4,
            }
        )
        == "Rebounding forward"
    )
    assert name_archetype({"r_fg3a_share": 0.3, "r_fga_per_min": -0.8}) == "Spot-up wing"
    assert name_archetype({}) == "Connective wing"


def test_describe_archetypes_inserts_description_after_label_and_dedupes():
    profiles = pl.DataFrame(
        {
            "archetype_label": [0, 1],
            "n": [10, 20],
            "r_fg3a_share": [0.3, 0.3],
            "r_fga_per_min": [-0.8, -0.8],
        }
    )

    described = describe_archetypes(profiles)

    assert described.columns[:3] == [
        "archetype_label",
        "archetype_description",
        "n",
    ]
    assert described["archetype_description"].to_list() == [
        "Spot-up wing",
        "Spot-up wing 2",
    ]
