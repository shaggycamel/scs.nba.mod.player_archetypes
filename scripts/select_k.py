"""Sweep k on held-out likelihood, then test stability across eras."""

import sys
from datetime import date

import polars as pl

from player_archetypes import cluster

X = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "design.parquet")
names = [c for c in X.columns if c.startswith("r_")]
cutoff = date(2024, 10, 1)

print(f"rows={X.height} features={len(names)} cutoff={cutoff}", flush=True)
scores = cluster.select_k(X, names, ks=range(3, 13), cutoff=cutoff)
with pl.Config(tbl_rows=20, tbl_width_chars=200):
    print(scores)
scores.write_parquet("k_scores.parquet")

best = scores.sort("heldout_loglik", descending=True)["k"][0]
print(f"\nbest k by held-out log-likelihood: {best}", flush=True)

stab = cluster.stability(
    X,
    names,
    k=best,
    cutoffs=[date(2016, 10, 1), date(2020, 10, 1), date(2024, 10, 1)],
)
print(stab)
stab.write_parquet("stability.parquet")
