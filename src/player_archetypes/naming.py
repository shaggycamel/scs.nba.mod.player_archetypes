"""Human-readable archetype descriptions derived from profile z-scores.

The rules use only the twelve style features that enter the clustering. Role
features (`r_min`, `r_start_rate`) are deliberately absent: a rule that reads
them would silently fall through to 0.0 now that clustering is style-only.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl


def name_archetype(row: Mapping[str, float]) -> str:
    """Name an archetype from its style z-score signature, by rule.

    The order matters: the most specific signatures are tested first. These
    thresholds were set against the fitted profiles and are the one piece of
    interpretation in the pipeline -- every number elsewhere is computed.
    """

    def z(column: str) -> float:
        value = row.get(column, 0.0)
        return float(value) if value is not None else 0.0

    oreb, dreb, blk = z("r_oreb_pct"), z("r_dreb_pct"), z("r_blk_per_min")
    three, fga = z("r_fg3a_share"), z("r_fga_per_min")
    ast, astr = z("r_ast_pct"), z("r_ast_ratio")
    stl, ftr = z("r_stl_per_min"), z("r_ftr")

    if oreb > 1.2 and blk > 0.8 and three < -1.0:
        return "Rim-running center"
    if blk > 1.0 and oreb > 0.8 and three < -0.5:
        return "Rim-protecting big"
    if blk > 0.4 and oreb > 0.3 and ast < -0.8:
        return "Interior finisher"
    if ast > 1.5 and astr > 1.0:
        return "Playmaking guard"
    if astr > 1.0 and ast > 0.7 and three > 0.2:
        return "Secondary creator"
    if ast > 0.7 and fga > 0.9:
        return "Scoring wing"
    if stl > 0.6 and ftr < -0.5 and blk < -0.3:
        return "Perimeter defensive wing"
    if three > 0.6 and ast < -0.3:
        return "Floor spacer"
    if dreb > 0.6 and fga > 0.5 and three < 0.0:
        return "Rebounding forward"
    if three > 0.2 and fga < -0.5:
        return "Spot-up wing"
    return "Connective wing"


def describe_archetypes(profiles: pl.DataFrame) -> pl.DataFrame:
    """Add ``archetype_description`` after ``archetype_label``.

    Duplicate names are disambiguated with a numeric suffix, in label order.
    The rules above are intended to be unique on the current taxonomy; the
    suffix is a fallback, not a role or depth order.
    """
    descriptions: list[str] = []
    used: dict[str, int] = {}
    for row in profiles.iter_rows(named=True):
        base = name_archetype(row)
        used[base] = used.get(base, 0) + 1
        descriptions.append(base if used[base] == 1 else f"{base} {used[base]}")

    result = profiles.with_columns(pl.Series("archetype_description", descriptions))
    if "archetype_label" in result.columns:
        columns = ["archetype_label", "archetype_description"] + [
            column
            for column in result.columns
            if column not in {"archetype_label", "archetype_description"}
        ]
        result = result.select(columns)
    return result
