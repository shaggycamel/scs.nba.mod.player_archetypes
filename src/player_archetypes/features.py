"""Archetype features: what a player does, measured from prior games only.

Design rules, all of them deliberate:

* Rate, not volume. Archetype is a *mix* of activities, so counting stats are
  normalised per minute and shot types are expressed as shares of the player's
  own attempts. Two players with the same shot diet belong together whether
  they play 12 minutes or 36.
* No usage. `usg_pct` / `e_usg_pct` are the downstream target and never enter
  the inputs.
* No efficiency. `fg_pct`, `ts_pct`, `efg_pct` measure how *well* a player does
  something, not what they do. A cold-shooting spot-up shooter is still a
  spot-up shooter. Pass `include_efficiency=True` to test that choice.
* No team context. Ratings, `pace` and `poss` describe the five on the floor.
* No role. `start_rate` and `min` describe how much a player is trusted, not
  what he does. They live in `ROLE_FEATURES` and are off by default; pass
  `include_role=True` only for the ablation.
* Strictly prior games. Every feature at game g uses games 1..g-1 for that
  player, so a cluster assignment is knowable before tip-off.
"""

from __future__ import annotations

import polars as pl

# Style: what a player does on the floor. This is the clustering substance.
# Role and volume are deliberately absent; they belong in the minutes and
# absorption models, where they answer a different question.
STYLE_FEATURES: tuple[str, ...] = (
    "fg3a_share",  # three-point rate: how far out the player operates
    "ftr",  # free-throw rate: contact / rim pressure
    "fga_per_min",  # shot-seeking
    "ast_per_min",  # playmaking volume
    "ast_pct",  # playmaking share of teammate makes
    "ast_ratio",
    "tov_pct",
    "oreb_pct",  # offensive glass: a big-man marker
    "dreb_pct",
    "stl_per_min",
    "blk_per_min",  # rim protection
    "pf_per_min",
)

# Role, not style. Kept as a separate block so the ablation is explicit and the
# production feature list cannot drift back into conflating the two.
ROLE_FEATURES: tuple[str, ...] = ("start_rate", "min")

EFFICIENCY_FEATURES: tuple[str, ...] = ("fg_pct", "fg3_pct", "ft_pct", "ts_pct", "efg_pct")


def per_game(box: pl.DataFrame) -> pl.DataFrame:
    """Derive single-game archetype rates from a played-only box score frame."""
    safe_min = pl.when(pl.col("min") > 0).then(pl.col("min")).otherwise(None)
    has_fga = pl.col("fga").is_not_null() & (pl.col("fga") > 0)
    return box.with_columns(
        # A game with no field-goal attempt has a zero three-point share and a
        # zero free-throw rate, not a missing value. Leaving these null quietly
        # shortens the rolling window.
        fg3a_share=pl.when(has_fga).then(pl.col("fg3_a") / pl.col("fga")).otherwise(0.0),
        ftr=pl.when(has_fga).then(pl.col("fta") / pl.col("fga")).otherwise(0.0),
        fga_per_min=pl.col("fga") / safe_min,
        ast_per_min=pl.col("ast") / safe_min,
        stl_per_min=pl.col("stl") / safe_min,
        blk_per_min=pl.col("blk") / safe_min,
        pf_per_min=pl.col("pf") / safe_min,
        # A null `start_position` means the player came off the bench. Without
        # the fill_null, `is_in` returns null, so a bench game is missing rather
        # than zero and the rolling rate only averages starts.
        start_rate=pl.col("start_position")
        .is_in(["G", "F", "C"])
        .fill_null(False)
        .cast(pl.Float64),
    )


def rolling(
    per_game_df: pl.DataFrame,
    window: int = 20,
    min_periods: int = 5,
    include_efficiency: bool = False,
    include_role: bool = False,
    half_life: float | None = None,
) -> pl.DataFrame:
    """Trailing per-player averages over the previous `window` games.

    The shift(1) inside each expression is what makes this leakage-free: the
    value at game g never sees game g. Rows with fewer than `min_periods`
    prior games come back null and are dropped by the caller.

    Style only by default. `include_role=True` adds `start_rate` and `min` for
    the explicit role/style ablation; production clustering must not use them.

    A null trailing value is filled with the player's last computed value. That
    is backwards-looking, so it cannot leak, and it keeps a player with one
    short-handed window in the assignment rather than dropping the row.

    `half_life` switches from a flat mean to an exponentially weighted mean,
    which tracks a mid-season archetype change faster.
    """
    cols = list(STYLE_FEATURES)
    if include_efficiency:
        cols += list(EFFICIENCY_FEATURES)
    if include_role:
        cols += list(ROLE_FEATURES)

    df = per_game_df.sort(["player_id", "game_date", "game_id"])
    if half_life is None:
        aggs = [
            pl.col(c)
            .shift(1)
            .rolling_mean(window_size=window, min_samples=min_periods)
            .over("player_id")
            .alias(f"r_{c}")
            for c in cols
        ]
    else:
        aggs = [
            pl.col(c)
            .shift(1)
            .ewm_mean(half_life=half_life, min_samples=min_periods, ignore_nulls=True)
            .over("player_id")
            .alias(f"r_{c}")
            for c in cols
        ]
    aggs.append(pl.col("game_id").shift(1).cum_count().over("player_id").alias("prior_games"))
    # The last game that fed the features, before any fallback fill.
    aggs.append(pl.col("game_date").shift(1).over("player_id").alias("features_through"))
    out = df.with_columns(aggs)
    return out.with_columns([pl.col(f"r_{c}").forward_fill().over("player_id") for c in cols])


def feature_names(include_efficiency: bool = False, include_role: bool = False) -> list[str]:
    cols = list(STYLE_FEATURES)
    if include_efficiency:
        cols += list(EFFICIENCY_FEATURES)
    if include_role:
        cols += list(ROLE_FEATURES)
    return [f"r_{c}" for c in cols]


def design_matrix(
    rolled: pl.DataFrame,
    include_efficiency: bool = False,
    include_role: bool = False,
    min_prior_games: int = 5,
) -> tuple[pl.DataFrame, list[str]]:
    """Rows ready for clustering: complete trailing features, enough history."""
    names = feature_names(include_efficiency, include_role)
    keep = ["player_id", "player_name", "game_id", "game_date", "season", "team_abbreviation"]
    out = (
        rolled.select([*keep, *names, "prior_games", "features_through"])
        .filter(pl.col("prior_games") >= min_prior_games)
        .drop_nulls(names)
    )
    return out, names


# ---------------------------------------------------------------------------
# Usage history. These are baselines for the downstream model and legitimate
# inputs to it, but they must never enter the clustering features: a archetype is
# what a player does, and usage is the thing we are trying to predict.
# ---------------------------------------------------------------------------

USAGE_HISTORY: tuple[str, ...] = (
    "usg_last5",
    "usg_last20",
    "usg_season_to_date",
    "min_last5",
)


def usage_history(per_game_df: pl.DataFrame) -> pl.DataFrame:
    """Prior-game usage and minutes, per player. All shift(1): never self-aware."""
    df = per_game_df.sort(["player_id", "game_date", "game_id"])
    return df.with_columns(
        usg_last5=pl.col("usg_pct")
        .shift(1)
        .rolling_mean(window_size=5, min_samples=2)
        .over("player_id"),
        usg_last20=pl.col("usg_pct")
        .shift(1)
        .rolling_mean(window_size=20, min_samples=5)
        .over("player_id"),
        usg_season_to_date=pl.col("usg_pct").shift(1).cum_sum().over(["player_id", "season"])
        / pl.col("usg_pct").shift(1).cum_count().over(["player_id", "season"]),
        min_last5=pl.col("min")
        .shift(1)
        .rolling_mean(window_size=5, min_samples=2)
        .over("player_id"),
    )
