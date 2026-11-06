"""The full experiment: choose k, test stability, then test whether archetypes help.

Writes every result to artifacts/ as parquet/json for the HTML report.

Usage: uv run python -m scripts.experiment
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from player_archetypes import cluster, features, naming
from player_archetypes.data import load

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

CUTOFF = date(2024, 10, 1)       # fit archetypes before this, evaluate after
FIT_SAMPLE = 120_000             # rows used to fit a mixture (14 dims: ample)
SEED = 0
KS = list(range(3, 13))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
log("loading box scores")
box = load.box_scores()
pg = features.usage_history(features.per_game(box))

log("building rolling archetype features (flat 20-game window)")
rolled = features.rolling(pg, window=20, min_periods=5)
X, names = features.design_matrix(rolled)
log(f"design matrix: {X.height:,} rows x {len(names)} features")

train_mask = pl.col("game_date") < CUTOFF
train = X.filter(train_mask)
held = X.filter(~train_mask)
fit_rows = train.sample(min(FIT_SAMPLE, train.height), seed=SEED)
log(f"fit={fit_rows.height:,} train={train.height:,} heldout={held.height:,}")

scaler = StandardScaler().fit(fit_rows.select(names).to_numpy())
x_fit = scaler.transform(fit_rows.select(names).to_numpy())
x_held = scaler.transform(held.select(names).to_numpy())


# ---------------------------------------------------------------------------
# 1. How many archetypes?
# ---------------------------------------------------------------------------
log("sweeping k")
rows = []
for k in KS:
    t0 = time.time()
    gmm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=SEED,
        n_init=2, max_iter=300,
    ).fit(x_fit)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=4).fit(x_fit)
    shares = np.bincount(gmm.predict(x_held), minlength=k) / x_held.shape[0]
    rows.append({
        "k": k,
        "bic": float(gmm.bic(x_fit)),
        "fit_loglik": float(gmm.score(x_fit)),
        "heldout_loglik": float(gmm.score(x_held)),
        "min_cluster_share": float(shares.min()),
        "kmeans_inertia": float(km.inertia_),
        "secs": round(time.time() - t0, 1),
    })
    log(f"  k={k} heldout={rows[-1]['heldout_loglik']:.4f} "
        f"min_share={rows[-1]['min_cluster_share']:.3f} ({rows[-1]['secs']}s)")

ksweep = pl.DataFrame(rows)
ksweep.write_parquet(ART / "k_sweep.parquet")

# Prefer the knee, not the maximum: held-out likelihood keeps creeping up, so
# take the smallest k within 1% of the best and with no degenerate cluster.
best_ll = ksweep["heldout_loglik"].max()
viable = ksweep.filter(
    (pl.col("heldout_loglik") >= best_ll - 0.01 * abs(best_ll))
    & (pl.col("min_cluster_share") >= 0.02)
)
K = int(viable["k"].min()) if viable.height else int(
    ksweep.sort("heldout_loglik", descending=True)["k"][0]
)
log(f"chosen k={K}")


# ---------------------------------------------------------------------------
# 2. Do the archetypes survive refitting on a different era?
# ---------------------------------------------------------------------------
log("stability across eras")
eras = [date(2014, 10, 1), date(2018, 10, 1), date(2021, 10, 1), CUTOFF]
ref = X.filter(pl.col("game_date") >= CUTOFF)
x_ref = scaler.transform(ref.select(names).to_numpy())
labels = {}
for era in eras:
    sub = X.filter(pl.col("game_date") < era)
    s = sub.sample(min(FIT_SAMPLE, sub.height), seed=SEED)
    sc = StandardScaler().fit(s.select(names).to_numpy())
    g = GaussianMixture(n_components=K, covariance_type="full",
                        random_state=SEED, n_init=2, max_iter=300).fit(
        sc.transform(s.select(names).to_numpy()))
    labels[era] = g.predict(sc.transform(ref.select(names).to_numpy()))

stab = pl.DataFrame([
    {"cutoff_a": str(a), "cutoff_b": str(b),
     "ari": float(adjusted_rand_score(labels[a], labels[b]))}
    for i, a in enumerate(eras) for b in eras[i + 1:]
])
stab.write_parquet(ART / "stability.parquet")
log(f"ARI range {stab['ari'].min():.3f} - {stab['ari'].max():.3f}")

# Seed sensitivity at the chosen k.
seed_labels = []
for s in range(4):
    g = GaussianMixture(n_components=K, covariance_type="full", random_state=s,
                        n_init=2, max_iter=300).fit(x_fit)
    seed_labels.append(g.predict(x_ref))
seed_ari = pl.DataFrame([
    {"seed_a": i, "seed_b": j,
     "ari": float(adjusted_rand_score(seed_labels[i], seed_labels[j]))}
    for i in range(4) for j in range(i + 1, 4)
])
seed_ari.write_parquet(ART / "seed_stability.parquet")
log(f"seed ARI range {seed_ari['ari'].min():.3f} - {seed_ari['ari'].max():.3f}")


# ---------------------------------------------------------------------------
# 3. Archetype profiles and the comparison against listed position
# ---------------------------------------------------------------------------
log("fitting final model and profiling archetypes")
model = cluster.fit(fit_rows, names, k=K, cutoff=None, seed=SEED)
assigned = model.assign(X)
assigned.select(
    "player_id", "player_name", "game_id", "game_date", "season",
    "archetype_label", "archetype_confidence", "archetype_entropy",
    *[f"archetype_{i}" for i in range(K)],
).write_parquet(ART / "assignments.parquet")

naming.describe_archetypes(cluster.profile(assigned, names)).write_parquet(
    ART / "archetype_profiles.parquet"
)

recent = assigned.filter(pl.col("season") == "2025-26")
exemplars = (
    recent.group_by("archetype_label", "player_name")
    .agg(pl.col("archetype_confidence").mean().alias("conf"), pl.len().alias("games"))
    .filter(pl.col("games") >= 20)
    .sort(["archetype_label", "conf"], descending=[False, True])
)
exemplars.write_parquet(ART / "exemplars.parquet")

pos = load.listed_positions()
joined = recent.join(pos, on=["season", "player_id"], how="inner")
joined.group_by("position", "archetype_label").agg(pl.len().alias("n")).write_parquet(
    ART / "position_crosstab.parquet")

spread = (
    joined.group_by("position", "archetype_label").agg(pl.len().alias("n"))
    .with_columns(share=pl.col("n") / pl.col("n").sum().over("position"))
    .group_by("position")
    .agg((-(pl.col("share") * pl.col("share").log()).sum()).alias("entropy"),
         pl.col("n").sum().alias("n"))
    .with_columns(archetypes_spanned=pl.col("entropy").exp().round(2))
    .sort("n", descending=True)
)
spread.write_parquet(ART / "position_spread.parquet")
log(f"position spread:\n{spread}")


# ---------------------------------------------------------------------------
# 4. The test that matters: do archetypes improve usage prediction?
# ---------------------------------------------------------------------------
log("downstream usage model")
usage_cols = list(features.USAGE_HISTORY)
archetype_cols = [f"archetype_{i}" for i in range(K)]

panel = (
    assigned.join(
        pg.select("player_id", "game_id", "usg_pct", *usage_cols),
        on=["player_id", "game_id"], how="inner")
    .drop_nulls(["usg_pct", *usage_cols])
)
tr = panel.filter(pl.col("game_date") < CUTOFF)
te = panel.filter(pl.col("game_date") >= CUTOFF)
log(f"panel train={tr.height:,} test={te.height:,}")

y_tr, y_te = tr["usg_pct"].to_numpy(), te["usg_pct"].to_numpy()
results = []


def score(label: str, pred: np.ndarray) -> None:
    results.append({
        "model": label,
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(np.sqrt(np.mean((y_te - pred) ** 2))),
    })
    log(f"  {label:38s} MAE={results[-1]['mae']:.5f} RMSE={results[-1]['rmse']:.5f}")


# Baselines first, as CLAUDE.md demands.
score("baseline: global mean", np.full_like(y_te, y_tr.mean()))
score("baseline: last-5 usage", te["usg_last5"].to_numpy())
score("baseline: last-20 usage", te["usg_last20"].to_numpy())
score("baseline: season-to-date usage", te["usg_season_to_date"].to_numpy())

feature_sets = {
    "usage history only": usage_cols,
    "usage history + archetype features": usage_cols + names,
    "usage history + soft archetypes": usage_cols + archetype_cols,
    "usage history + archetype features + soft archetypes": usage_cols + names + archetype_cols,
    "archetype features only": names,
}
try:
    import lightgbm as lgb
    have_lgb = True
except ImportError:
    have_lgb = False

for label, cols in feature_sets.items():
    a = tr.select(cols).to_numpy()
    b = te.select(cols).to_numpy()
    sc = StandardScaler().fit(a)
    ridge = Ridge(alpha=1.0).fit(sc.transform(a), y_tr)
    score(f"ridge: {label}", ridge.predict(sc.transform(b)))
    if have_lgb:
        gbm = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            random_state=SEED, verbose=-1,
        ).fit(a, y_tr)
        score(f"lightgbm: {label}", gbm.predict(b))

res = pl.DataFrame(results).sort("mae")
res.write_parquet(ART / "usage_models.parquet")

# Feature importance from the best full model, for the write-up.
if have_lgb:
    cols = usage_cols + names + archetype_cols
    gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                            random_state=SEED, verbose=-1).fit(
        tr.select(cols).to_numpy(), y_tr)
    pl.DataFrame({"feature": cols, "gain": gbm.booster_.feature_importance("gain")}) \
        .sort("gain", descending=True).write_parquet(ART / "feature_importance.parquet")

meta = {
    "k": K,
    "cutoff": str(CUTOFF),
    "design_rows": X.height,
    "n_features": len(names),
    "features": names,
    "players": int(X["player_id"].n_unique()),
    "date_min": str(X["game_date"].min()),
    "date_max": str(X["game_date"].max()),
    "panel_train": tr.height,
    "panel_test": te.height,
    "box_played_rows": box.height,
    "mean_confidence": float(assigned["archetype_confidence"].mean()),
    "frac_ambiguous": float((assigned["archetype_confidence"] < 0.6).mean()),
}
(ART / "meta.json").write_text(json.dumps(meta, indent=2))
log("done")
print(res)
