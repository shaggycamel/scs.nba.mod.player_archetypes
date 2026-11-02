"""Validate the roles against play-type data they were never shown.

This is the test the clustering did not get to train for. `statyx.play_types`
records how each player's offence is actually generated -- spot-up, iso,
pick-and-roll handler, roll man, post-up, cut, hand-off, off-screen, transition,
offensive board. None of it entered the features, the fit, or the k choice.

The question: do the roles predict that mix better than listed position does?
That is the comparison that matters, because position is the thing the roles are
meant to replace.

Caveat stated plainly in the output: play_types is a whole-season aggregate for
2025-26 and the role responsibilities average over that same season, so the
headline numbers measure concurrent validity, not forecasting. The `early`
variant restricts roles to each player's first 20 games of the season to show
what survives when most of the overlap is removed.

Usage: uv run python -m scripts.validate_playtypes
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from player_archetypes.data import load

ART = Path("artifacts")
SEED = 0
SEASON = "2025-26"
KINDS = [
    "spot_up", "iso", "transition", "p_and_r_ball_handler", "p_and_r_roll_man",
    "cut", "hand_off", "off_screen", "post_up", "o_board",
]
TARGETS = [f"pt_share_{k}" for k in KINDS]


def log(m: str) -> None:
    print(m, flush=True)


meta = json.loads((ART / "meta.json").read_text())
K = meta["k"]
role_cols = [f"role_{i}" for i in range(K)]

MIN_ATTEMPTS = 25   # below this a player's play-type shares are mostly noise

assign = pl.read_parquet(ART / "assignments.parquet").filter(pl.col("season") == SEASON)
pt_all = load.play_types().filter(pl.col("season") == SEASON)
pos = load.listed_positions().filter(pl.col("season") == SEASON)
log(f"assignments(2025-26)={assign.height:,}  play_types={pt_all.height}  positions={pos.height}")

# Defect 5: three play-type columns are identically zero across every player in
# the dump. Roll man and post-up plainly occur, so this is a collection gap
# upstream, not a fact about the season. Drop them rather than score NaNs.
dead = [t for t in TARGETS if pt_all[t].std() == 0]
if dead:
    log(f"dropping all-zero play types (upstream gap): "
        f"{', '.join(d.removeprefix('pt_share_') for d in dead)}")
TARGETS = [t for t in TARGETS if t not in dead]

pt = pt_all.filter(pl.col("pt_total_attempts") >= MIN_ATTEMPTS)
log(f"play-type players with >= {MIN_ATTEMPTS} attempts: {pt.height} of {pt_all.height} "
    f"(median total attempts {pt_all['pt_total_attempts'].median():.0f})")


def player_roles(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("player_id")
        .agg([pl.col(c).mean().alias(c) for c in role_cols]
             + [pl.len().alias("games"),
                pl.col("role_label").mode().first().alias("modal_role")])
        .filter(pl.col("games") >= 20)
    )


variants = {
    "full season": player_roles(assign),
    "first 20 games": player_roles(
        assign.sort(["player_id", "game_date"])
        .with_columns(n=pl.col("game_id").cum_count().over("player_id"))
        .filter(pl.col("n") <= 20)
    ),
}

rows, prof_rows = [], []
for vname, roles in variants.items():
    panel = (
        roles.join(pt, on="player_id", how="inner")
        .join(pos.select("player_id", "position"), on="player_id", how="left")
        .drop_nulls(TARGETS)
    )
    log(f"\n[{vname}] matched players: {panel.height}")
    if panel.height < 60:
        log("  too few to evaluate")
        continue

    pos_dummies = panel.select("position").to_dummies(drop_first=True)
    blocks = {
        "roles (soft)": panel.select(role_cols).to_numpy(),
        "listed position": pos_dummies.to_numpy().astype(float),
        "roles + position": np.hstack(
            [panel.select(role_cols).to_numpy(), pos_dummies.to_numpy().astype(float)]
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for bname, X in blocks.items():
        Xs = StandardScaler().fit_transform(X)
        r2s = []
        for t in TARGETS:
            y = panel[t].to_numpy()
            pred = cross_val_predict(
                RidgeCV(alphas=np.logspace(-2, 3, 20)), Xs, y, cv=cv)
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r2s.append(r2)
            rows.append({"variant": vname, "features": bname,
                         "play_type": t.removeprefix("pt_share_"), "cv_r2": r2})
        log(f"  {bname:18s} mean CV R2 = {np.nanmean(r2s):.4f}")

    if vname == "full season":
        # Per-role play-type signature, for the report heatmap.
        mu = panel.select(TARGETS).mean().to_numpy()[0]
        sd = panel.select(TARGETS).std().to_numpy()[0]
        g = (panel.group_by("modal_role")
             .agg([pl.col(t).mean().alias(t) for t in TARGETS] + [pl.len().alias("n")])
             .sort("modal_role"))
        for r in g.iter_rows(named=True):
            out = {"modal_role": r["modal_role"], "n": r["n"]}
            for i, t in enumerate(TARGETS):
                out[t] = (r[t] - mu[i]) / sd[i] if sd[i] else 0.0
            prof_rows.append(out)

res = pl.DataFrame(rows)
res.write_parquet(ART / "playtype_validation.parquet")
if prof_rows:
    pl.DataFrame(prof_rows).write_parquet(ART / "playtype_role_profiles.parquet")

log("\n=== mean CV R-squared by feature block ===")
summary = (res.with_columns(
               pl.when(pl.col("cv_r2").is_nan()).then(None).otherwise(pl.col("cv_r2")).alias("cv_r2"))
           .group_by("variant", "features")
           .agg(pl.col("cv_r2").mean().alias("mean_cv_r2"),
                pl.col("cv_r2").count().alias("n_play_types"))
           .sort(["variant", "mean_cv_r2"], descending=[False, True]))
print(summary)
summary.write_parquet(ART / "playtype_summary.parquet")

log("\n=== per play type, full season ===")
with pl.Config(tbl_rows=40, tbl_width_chars=150):
    print(res.filter(pl.col("variant") == "full season")
          .pivot(on="features", index="play_type", values="cv_r2")
          .with_columns(pl.exclude("play_type").round(3)))
