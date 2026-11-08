"""Data audit: the defects the loaders work around, reproduced from raw parquet.

Run: uv run python scripts/audit.py
"""

from __future__ import annotations

import polars as pl

from player_archetypes.data.ids import DATA, game_map, nba_schedule, player_map


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


rule("1. DNPs are encoded as zero usage, not missing")
box = pl.scan_parquet(DATA / "nba" / "player_box_score.parquet")
dnp = box.filter(pl.col("min").is_null())
played = box.filter(pl.col("min").is_not_null() & (pl.col("min") > 0))
print(
    box.select(
        pl.len().alias("all_rows"),
        pl.col("usg_pct").mean().round(4).alias("usg_mean_naive"),
    )
    .collect()
    .to_dicts()[0]
)
print(
    dnp.select(
        pl.len().alias("dnp_rows"),
        pl.col("usg_pct").mean().round(6).alias("dnp_usg_mean"),
        pl.col("usg_pct").null_count().alias("dnp_usg_nulls"),
    )
    .collect()
    .to_dicts()[0]
)
print(
    played.select(
        pl.len().alias("played_rows"),
        pl.col("usg_pct").mean().round(4).alias("usg_mean_played"),
    )
    .collect()
    .to_dicts()[0]
)
print("-> a naive mean is biased low by ~18% of rows that never played")

rule("2. statyx.advanced_stats is duplicated and its season label is wrong")
adv = pl.read_parquet(DATA / "statyx" / "advanced_stats.parquet")
key = ["player_id", "game_id"]
print(
    {
        "raw_rows": adv.height,
        "unique_rows": adv.unique().height,
        "unique_keys": adv.select(key).unique().height,
    }
)
dups = adv.group_by(key).agg(
    pl.len().alias("n"),
    pl.col("touches").n_unique().alias("touch_vals"),
    pl.col("usage_percentage").n_unique().alias("usage_vals"),
)
print("copies per key:", dups["n"].value_counts().sort("n").to_dicts())
print(
    {
        "keys_conflicting_touches": dups.filter(pl.col("touch_vals") > 1).height,
        "keys_conflicting_usage": dups.filter(pl.col("usage_vals") > 1).height,
    }
)
print(
    adv.group_by("season")
    .agg(
        pl.col("game_date").min().alias("d_min"),
        pl.col("game_date").max().alias("d_max"),
        pl.col("game_id").n_unique().alias("games"),
    )
    .sort("season")
)
print("-> both season labels span the same dates and games: the label is unusable")
print("-> 'period' values:", adv["period"].unique().to_list(), "(whole-game rows)")

rule("3. nba and statyx share no identifiers")
bx_ids = box.select("game_id", "player_id").head(2).collect().to_dicts()
adv_ids = adv.select("game_id", "player_id").head(2).to_dicts()
print("nba   :", bx_ids)
print("statyx:", adv_ids)
pm, gm = player_map(), game_map()
sx_sched = pl.read_parquet(DATA / "statyx" / "schedule.parquet")
print(
    {
        "player_map_pairs": pm.height,
        "statyx_games": sx_sched.height,
        "games_mapped": gm.height,
        "games_unmapped": sx_sched.height - gm.height,
    }
)
print("-> players bridge via util.player_id_map_vw; games are rebuilt on date+abbr")

rule("4. Listed position is coarse where it exists at all")
print(box.group_by("start_position").agg(pl.len().alias("n")).sort("n", descending=True).collect())
info = pl.read_parquet(DATA / "nba" / "player_info.parquet")
print(info.group_by("position").agg(pl.len().alias("n")).sort("n", descending=True))
print(
    "-> box score offers only G/F/C; player_info adds hyphenates. Neither",
    "\n   distinguishes a stretch five from a post-up five.",
)

rule("5. Coverage asymmetry across schemas")
sched = nba_schedule().select(pl.col("nba_game_id").alias("game_id"), "season")
print(
    box.join(sched.lazy(), on="game_id", how="left")
    .group_by("season")
    .agg(pl.len().alias("rows"))
    .sort("season")
    .collect()
    .tail(6)
)
for name in ["play_types", "advanced_stats", "game_stats", "schedule"]:
    df = pl.read_parquet(DATA / "statyx" / f"{name}.parquet")
    print(f"statyx.{name:16s} seasons={sorted(df['season'].unique().to_list())}")
