"""Fit k roles, name them from their feature profiles, and compare to position.

Usage: uv run python -m scripts.role_report <design.parquet> <k> [cutoff]
"""
from __future__ import annotations

import sys
from datetime import date

import polars as pl

from player_archetypes import cluster
from player_archetypes.data.load import listed_positions

design_path = sys.argv[1]
k = int(sys.argv[2])
cutoff = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2024, 10, 1)

X = pl.read_parquet(design_path)
names = [c for c in X.columns if c.startswith("r_")]
print(f"rows={X.height} k={k} cutoff={cutoff}\n", flush=True)

model = cluster.fit(X, names, k=k, cutoff=cutoff)
assigned = model.assign(X)

print("=" * 78)
print("ROLE PROFILES (feature means as z-scores; |z| > 0.5 is the role's signature)")
print("=" * 78)
prof = cluster.profile(assigned, names)
with pl.Config(tbl_cols=20, tbl_width_chars=240, tbl_rows=30):
    print(prof)

print("\n" + "=" * 78)
print("WHAT DEFINES EACH ROLE (top |z| features)")
print("=" * 78)
for row in prof.iter_rows(named=True):
    label = row["role_label"]
    feats = sorted(
        ((c, row[c]) for c in names), key=lambda kv: abs(kv[1]), reverse=True
    )[:5]
    desc = ", ".join(f"{c.removeprefix('r_')} {v:+.2f}" for c, v in feats)
    print(f"role {label} (n={row['n']:>6,}): {desc}")

print("\n" + "=" * 78)
print("EXEMPLARS: highest-confidence player-seasons per role (2025-26)")
print("=" * 78)
recent = assigned.filter(pl.col("season") == "2025-26")
for label in range(k):
    top = (
        recent.filter(pl.col("role_label") == label)
        .group_by("player_name")
        .agg(
            pl.col("role_confidence").mean().alias("conf"),
            pl.len().alias("games"),
        )
        .filter(pl.col("games") >= 20)
        .sort("conf", descending=True)
        .head(6)
    )
    who = ", ".join(top["player_name"].to_list())
    print(f"role {label}: {who}")

print("\n" + "=" * 78)
print("VS LISTED POSITION: does the clustering actually say something new?")
print("=" * 78)
pos = listed_positions()
joined = recent.join(pos, on=["season", "player_id"], how="inner")
ct = (
    joined.group_by("position", "role_label")
    .agg(pl.len().alias("n"))
    .pivot(on="role_label", index="position", values="n")
    .fill_null(0)
)
with pl.Config(tbl_cols=20, tbl_width_chars=200):
    print(ct)

# How spread is each listed position across roles? High entropy means the
# position label is throwing away real distinctions.
shares = (
    joined.group_by("position", "role_label").agg(pl.len().alias("n"))
    .with_columns(share=pl.col("n") / pl.col("n").sum().over("position"))
)
ent = (
    shares.group_by("position")
    .agg(
        (-(pl.col("share") * pl.col("share").log()).sum()).alias("entropy_nats"),
        pl.col("n").sum().alias("n"),
    )
    .with_columns(
        roles_spanned=pl.col("entropy_nats").exp().round(2),
    )
    .sort("n", descending=True)
)
print("\nEffective number of roles each listed position spans (exp of entropy):")
print(ent)

print("\n" + "=" * 78)
print("SOFTNESS: how many player-games are genuinely split between roles?")
print("=" * 78)
print(
    assigned.select(
        pl.col("role_confidence").mean().round(3).alias("mean_confidence"),
        (pl.col("role_confidence") < 0.6).mean().round(3).alias("frac_ambiguous"),
        pl.col("role_entropy").mean().round(3).alias("mean_entropy"),
    )
)
print("-> frac_ambiguous is the share a hard label would misrepresent")
