"""Human-readable archetype descriptions derived from profile z-scores."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl


def name_archetype(row: Mapping[str, float]) -> str:
    """Name an archetype from its own z-score signature, by rule.

    The order matters: the most specific signatures are tested first. These
    thresholds were set against the fitted profiles and are the one piece of
    interpretation in the pipeline -- every number elsewhere is computed.
    """

    def z(column: str) -> float:
        value = row.get(column, 0.0)
        return float(value) if value is not None else 0.0

    oreb, blk, three = z("r_oreb_pct"), z("r_blk_per_min"), z("r_fg3a_share")
    ast, astr = z("r_ast_pct"), z("r_ast_ratio")
    mins, start, fga = z("r_min"), z("r_start_rate"), z("r_fga_per_min")

    if oreb > 1.0 and three < -1.0:
        return "Rim-running center"
    if ast > 1.0 and mins > 0.8 and fga > 0.5:
        return "Primary creator"
    if ast > 1.0:
        return "Playmaking guard"
    if oreb > 0.4 and blk > 0.4 and ast < -0.5:
        return "Interior finisher"
    if mins < -1.0:
        return "Deep reserve"
    if start > 0.8:
        return "Starting big" if ast < 0 else "Starting wing"
    if three > 0.6 and start < -0.5:
        return "Bench shooter"
    if three > 0.6:
        return "Floor spacer"
    if astr > 0.6 and fga < -0.4:
        return "Low-usage connector"
    if start < -0.8:
        return "Bench wing"
    return "Connective wing"


def describe_archetypes(profiles: pl.DataFrame) -> pl.DataFrame:
    """Add ``archetype_description`` after ``archetype_label``.

    Duplicate names are disambiguated with a numeric suffix, in label order.
    """
    descriptions: list[str] = []
    used: dict[str, int] = {}
    for row in profiles.iter_rows(named=True):
        base = name_archetype(row)
        used[base] = used.get(base, 0) + 1
        descriptions.append(base if used[base] == 1 else f"{base} {used[base]}")

    result = profiles.with_columns(
        pl.Series("archetype_description", descriptions)
    )
    if "archetype_label" in result.columns:
        columns = ["archetype_label", "archetype_description"] + [
            column
            for column in result.columns
            if column not in {"archetype_label", "archetype_description"}
        ]
        result = result.select(columns)
    return result
