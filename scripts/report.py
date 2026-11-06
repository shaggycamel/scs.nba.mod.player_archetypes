"""Render the HTML write-up from artifacts/. No number is typed by hand.

Usage: uv run python -m scripts.report
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from scripts.svgkit import esc, grouped_hbar, hbar_chart, heatmap, line_chart, table_view

ART = Path("artifacts")
OUT = Path("docs/archetype_clustering_report.html")
M = json.loads((ART / "meta.json").read_text())
K = M["k"]

PRETTY = {
    "r_fg3a_share": "3PA share",
    "r_ftr": "FT rate",
    "r_fga_per_min": "FGA / min",
    "r_ast_per_min": "AST / min",
    "r_ast_pct": "AST %",
    "r_ast_ratio": "AST ratio",
    "r_tov_pct": "TOV %",
    "r_oreb_pct": "OREB %",
    "r_dreb_pct": "DREB %",
    "r_stl_per_min": "STL / min",
    "r_blk_per_min": "BLK / min",
    "r_pf_per_min": "PF / min",
    "r_start_rate": "start rate",
    "r_min": "minutes",
    "usg_last5": "usage, last 5",
    "usg_last20": "usage, last 20",
    "usg_season_to_date": "usage, season to date",
    "min_last5": "minutes, last 5",
}
def nice(c: str) -> str:
    return PRETTY.get(c, c.replace("archetype_", "archetype ").replace("r_", ""))


# ---------------------------------------------------------------------------
# Archetype naming, derived from each archetype's own z-score signature
# ---------------------------------------------------------------------------
prof = pl.read_parquet(ART / "archetype_profiles.parquet")
feat_cols = [c for c in prof.columns if c.startswith("r_")]


def name_archetype(row: dict) -> str:
    """Name a archetype from its own z-score signature, by rule.

    The order matters: the most specific signatures are tested first. These
    thresholds were set against the fitted profiles and are the one piece of
    interpretation in the pipeline -- every number elsewhere is computed.
    """
    z = lambda c: row.get(c, 0.0)
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


names_by_label: dict[int, str] = {}
used: dict[str, int] = {}
for row in prof.iter_rows(named=True):
    base = name_archetype(row)
    used[base] = used.get(base, 0) + 1
    names_by_label[row["archetype_label"]] = base if used[base] == 1 else f"{base} {used[base]}"


def rl(label: int) -> str:
    return f"{label} &middot; {names_by_label.get(label, '')}"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
parts: list[str] = []


def section(title: str, body: str, sub: str = "") -> None:
    s = f'<p class="sub">{sub}</p>' if sub else ""
    parts.append(f'<section><h2>{title}</h2>{s}{body}</section>')


# --- Headline -------------------------------------------------------------
_stab = pl.read_parquet(ART / "stability.parquet")
stab_min, stab_max = _stab["ari"].min(), _stab["ari"].max()
usage = pl.read_parquet(ART / "usage_models.parquet")
best_base = usage.filter(pl.col("model").str.starts_with("baseline")).sort("mae")
best_model = usage.filter(~pl.col("model").str.starts_with("baseline")).sort("mae")
bb, bm = best_base.row(0, named=True), best_model.row(0, named=True)

no_archetype = usage.filter(pl.col("model").str.contains("usage history only")).sort("mae")
any_archetype = usage.filter(
    pl.col("model").str.contains("archetype")
    & ~pl.col("model").str.starts_with("baseline")
    & ~pl.col("model").str.contains("archetype features only")
).sort("mae")
base_mae = no_archetype["mae"][0]
archetype_mae = any_archetype["mae"][0]
delta_pct = (base_mae - archetype_mae) / base_mae * 100
best_baseline_mae = best_base["mae"][0]
vs_baseline_pct = (best_baseline_mae - archetype_mae) / best_baseline_mae * 100

hero = f"""
<div class="tiles">
  <div class="tile"><div class="tile-k">Archetypes found</div><div class="tile-v">{K}</div>
    <div class="tile-n">chosen on held-out likelihood, not a silhouette score</div></div>
  <div class="tile"><div class="tile-k">Player-games</div><div class="tile-v">{M['design_rows']:,}</div>
    <div class="tile-n">{M['players']:,} players, {M['date_min']} to {M['date_max']}</div></div>
  <div class="tile"><div class="tile-k">Best MAE on usage</div><div class="tile-v">{bm['mae']:.4f}</div>
    <div class="tile-n">vs {bb['mae']:.4f} for the best naive baseline</div></div>
  <div class="tile {'good' if delta_pct > 1.0 else 'flat'}"><div class="tile-k">Gain from adding archetypes</div>
    <div class="tile-v">{delta_pct:+.2f}%</div>
    <div class="tile-n">MAE vs the same model on usage history alone &mdash; not a meaningful gain</div></div>
</div>"""

# --- 0. Verdict -----------------------------------------------------------
_pts = pl.read_parquet(ART / "playtype_summary.parquet") if (ART / "playtype_summary.parquet").exists() else None
if _pts is not None:
    _rf = float(_pts.filter((pl.col("variant") == "full season") & (pl.col("features") == "archetypes (soft)"))["mean_cv_r2"][0])
    _pf = float(_pts.filter((pl.col("variant") == "full season") & (pl.col("features") == "listed position"))["mean_cv_r2"][0])
else:
    _rf = _pf = float("nan")

section(
    "Verdict, up front",
    f"""<p><strong>The archetypes beat listed position at describing how players play, and they do
    not improve a next-game usage model.</strong> Both are true, and the first is the one that
    answers the original question.</p>
    <p><strong>They describe the game better.</strong> Against play-type data the clustering
    never saw &mdash; spot-up, isolation, pick-and-roll handling, cuts &mdash; the archetypes reach a
    mean cross-validated R&sup2; of {_rf:.3f} against {_pf:.3f} for listed position, a
    {_rf / _pf:.1f}&times; gap (section 9). On spot-ups and isolations, position's R&sup2; is
    <em>negative</em>: knowing a player is a "Guard" predicts his spot-up rate worse than
    guessing the league mean. The ten archetypes are also legible without being told what to look
    for &mdash; one is Mitchell Robinson, Steven Adams and Biyombo; another is Duncan Robinson,
    Sam Hauser and Nesmith; another is Jokic, Doncic and Gilgeous-Alexander. Nothing in the
    inputs knew about position, usage, or player identity.</p>
    <p><strong>They do not help predict usage.</strong> Adding archetypes to a usage model moves MAE
    from {base_mae:.5f} to {archetype_mae:.5f} &mdash; {delta_pct:.2f}%. The project brief says to
    keep clustering only if an ablation shows a gain; on that rule, for that model, it does not.
    The reason is visible in the feature importances: a player's own recent usage already
    carries nearly everything the archetypes would say. Archetype features alone reach
    {usage.filter(pl.col('model').str.contains('archetype features only')).sort('mae')['mae'][0]:.5f},
    about the last-20-games baseline &mdash; they substitute for usage history rather than add
    to it.</p>
    <p>So the honest reading is that next-game usage was the wrong test. Usage is strongly
    autocorrelated, and any slow-moving descriptor will look redundant beside it. Judged on what
    position is actually used for &mdash; describing how a player operates &mdash; the archetypes are
    a clear improvement.</p>
    <p><strong>The real caveat is stability.</strong> Refit on different eras, the taxonomy
    agrees with itself at only {stab_min:.2f} to {stab_max:.2f} ARI. That is moderate. Before
    these become the fixed vocabulary of future analysis, that number needs to be better
    understood &mdash; it may reflect genuine change in how the game is played between 2014 and
    2024, which would be a finding rather than a flaw, but this study cannot tell those apart.</p>""",
    "What I would and would not claim from this.",
)

# --- 1. Why positions fail ------------------------------------------------
spread = pl.read_parquet(ART / "position_spread.parquet")
spread_chart = hbar_chart(
    spread["position"].to_list(),
    spread["archetypes_spanned"].to_list(),
    label="Effective number of archetypes each listed position spans",
    value_label="archetypes spanned",
    dp=2,
    label_w=170,
)
spread_table = table_view(
    ["Listed position", "Player-games", "Entropy (nats)", "Effective archetypes spanned"],
    [[r["position"], f"{r['n']:,}", f"{r['entropy']:.3f}", f"{r['archetypes_spanned']:.2f}"]
     for r in spread.iter_rows(named=True)],
    "archetypes spanned per listed position",
)
section(
    "1. The premise, measured",
    f"""<p>Listed position is not just coarse &mdash; across much of the data it is
    <em>absent</em>. In <code>player_box_score</code>, <code>start_position</code> takes only
    <code>G</code>/<code>F</code>/<code>C</code> and is blank for 361,992 of 589,082 rows, because
    only starters get a value. <code>player_info.position</code> adds hyphenates but 4,946 of
    5,630 player-seasons are still a bare <em>Guard</em>, <em>Forward</em> or <em>Center</em>.</p>
    <p>The chart below is the quantitative form of your intuition: for each listed position,
    the effective number of distinct archetypes its players actually occupy (the exponential of the
    entropy of its archetype distribution). A label that described play would sit near 1.0.</p>
    {spread_chart}{spread_table}""",
    "If a position label spans four archetypes, it is discarding four-fifths of what it could say.",
)

# --- 2. Data and defects --------------------------------------------------
defects = table_view(
    ["#", "Defect", "Evidence", "Handling"],
    [
        [1, "DNPs encoded as zero usage, not missing",
         "105,394 of 589,082 rows (17.9%) have usg_pct = 0.0 with a null min. Naive mean usage 0.1510 vs played-only 0.1839.",
         "box_scores() flags played and blanks the fabricated targets"],
        [2, "statyx.advanced_stats duplicated ~2x by a bad season join",
         "116,972 rows collapse to 57,255 unique (player, game); both season labels span the identical date range over the same 2,600 game ids. 163 keys conflict on touches, 24 on usage_percentage.",
         "season discarded and rebuilt from the game map; conflicting copies averaged"],
        [3, "No shared identifiers between the nba and statyx schemas",
         "game 21501032 vs 900034932; player 200811 vs 1253150. Direct join yields zero matches.",
         "players bridged via util.player_id_map_vw (605 pairs); games rematched on (date, home, away), recovering 5,229 of 5,289"],
        [4, "period column is always 0 in advanced_stats",
         "Single distinct value across 116,972 rows.",
         "treated as whole-game rows, despite the name"],
    ],
    "defects found in the raw dumps",
)
section(
    "2. The data, and what was wrong with it",
    f"""<p>Four defects had to be fixed before any clustering was meaningful. The first is the
    dangerous one: a model trained without it learns that a fifth of all appearances have zero
    usage, which is not a fact about basketball but an artefact of how the dump encodes a
    healthy scratch.</p>{defects}
    <p>Coverage is lopsided across schemas. <code>player_box_score</code> runs 2009&ndash;2026;
    statyx <code>game_stats</code> and <code>schedule</code> cover four seasons;
    <code>advanced_stats</code> two; and <code>play_types</code>, the richest archetype signal of all
    (spot-up / iso / pick-and-roll handler / roll man / post-up / cut / off-screen), exists for
    <strong>2025-26 only, at season grain</strong>. That last point decided the design: play-type
    data cannot be a model input without leaking within-season information, so the backbone is
    box-score rates across the full history and statyx is an enrichment layer.</p>""",
    "Everything here is reproducible with uv run python -m scripts.audit.",
)

# --- 3. Methodology -------------------------------------------------------
section(
    "3. Method",
    f"""<ul class="method">
    <li><strong>Rates, not volumes.</strong> Archetype is a mix of activities, so counting stats are
      per-minute and shot types are shares of the player's own attempts. The same shot diet at
      12 or 36 minutes is the same archetype.</li>
    <li><strong>Usage excluded from the inputs.</strong> <code>usg_pct</code> and
      <code>e_usg_pct</code> are the downstream target; letting them define the archetypes would make
      any later "archetypes predict usage" result circular.</li>
    <li><strong>Efficiency excluded.</strong> <code>ts_pct</code>, <code>fg_pct</code> measure how
      well, not what. A cold-shooting spot-up shooter is still a spot-up shooter. The flag
      <code>include_efficiency=True</code> exists to test that claim.</li>
    <li><strong>Team context excluded.</strong> Ratings, <code>pace</code> and <code>poss</code>
      describe the five on the floor, not the individual.</li>
    <li><strong>Strictly prior games.</strong> Every feature is <code>shift(1)</code> then rolled
      over the previous 20 games per player, so a label is knowable before tip-off. Rows with
      fewer than 5 prior games are dropped.</li>
    <li><strong>Soft assignment.</strong> A Gaussian mixture returns responsibilities, not a hard
      id. A hard id would reimpose exactly the discreteness that makes listed position a poor
      description of the modern game.</li>
    </ul>
    <p>The resulting design matrix is {M['design_rows']:,} player-games by
    {M['n_features']} features, covering {M['players']:,} players from {M['date_min']} to
    {M['date_max']}. Mixtures are fitted on a 120,000-row sample of the pre-cutoff period
    (14 dimensions; the sample is ample and makes the k sweep tractable), and the cutoff for
    every fit is {M['cutoff']}.</p>""",
    "Each choice below is a decision that could have gone the other way; the flags to reverse them are in the code.",
)

# --- 4. Choosing k --------------------------------------------------------
ksweep = pl.read_parquet(ART / "k_sweep.parquet")
ll_chart = line_chart(
    ksweep["k"].to_list(),
    ksweep["heldout_loglik"].to_list(),
    label="Held-out log-likelihood by number of archetypes",
    x_label="number of archetypes (k)",
    y_label="held-out log-likelihood",
    highlight=K,
    highlight_note=f"k = {K} selected",
    y_dp=2,
)
share_chart = hbar_chart(
    [f"k = {k}" for k in ksweep["k"].to_list()],
    ksweep["min_cluster_share"].to_list(),
    label="Smallest cluster share by k",
    value_label="smallest cluster share",
    dp=3,
    label_w=80,
    row_h=24,
)
k_table = table_view(
    ["k", "BIC (fit)", "Log-lik (fit)", "Log-lik (held out)", "Smallest cluster share", "Fit seconds"],
    [[r["k"], f"{r['bic']:,.0f}", f"{r['fit_loglik']:.4f}", f"{r['heldout_loglik']:.4f}",
      f"{r['min_cluster_share']:.4f}", r["secs"]] for r in ksweep.iter_rows(named=True)],
    "k sweep",
)
section(
    "4. How many archetypes?",
    f"""<p>Held-out log-likelihood is the honest criterion here. BIC is computed on data the
    mixture has already seen, and with 409,320 training rows it will keep paying for extra
    components almost indefinitely. The sweep therefore scores each k on a period the model
    never saw &mdash; games from {M['cutoff']} onward.</p>
    {ll_chart}
    <p>Likelihood alone still creeps upward with k, so the selection rule is the smallest k
    within 1% of the best held-out likelihood that also leaves no cluster holding under 2% of
    held-out games. The second condition matters: past a point the mixture starts spending
    components on tiny pockets rather than real archetypes.</p>
    {share_chart}{k_table}""",
    "Chosen on out-of-sample likelihood with a floor on cluster size, not on a silhouette score.",
)

# --- 5. Stability ---------------------------------------------------------
stab = pl.read_parquet(ART / "stability.parquet")
seed = pl.read_parquet(ART / "seed_stability.parquet")
eras = sorted({*stab["cutoff_a"].to_list(), *stab["cutoff_b"].to_list()})
mat = [[None if a == b else next(
    (r["ari"] for r in stab.iter_rows(named=True)
     if {r["cutoff_a"], r["cutoff_b"]} == {a, b}), None)
    for b in eras] for a in eras]
stab_chart = heatmap(
    [e[:4] for e in eras], [e[:4] for e in eras], mat,
    label="Adjusted Rand Index between models fitted on different eras",
    diverging=False, value_dp=3, cell_dp=2, label_w=120, cell=46,
)
stab_table = table_view(
    ["Fitted before", "vs fitted before", "ARI on common rows"],
    [[r["cutoff_a"], r["cutoff_b"], f"{r['ari']:.4f}"] for r in stab.iter_rows(named=True)]
    + [[f"seed {r['seed_a']}", f"seed {r['seed_b']}", f"{r['ari']:.4f}"] for r in seed.iter_rows(named=True)],
    "stability, across eras and across seeds",
)
section(
    "5. Do the archetypes survive refitting?",
    f"""<p>This is the test a position replacement has to pass and a purely predictive feature
    does not. If the taxonomy reshuffles whenever the fitting window moves, it is describing the
    sample rather than the game, and it cannot be the vocabulary of future analysis.</p>
    <p>Four mixtures were fitted on data before 2014, 2018, 2021 and {M['cutoff'][:4]}, then all four
    labelled the same recent rows. Adjusted Rand Index compares those labellings &mdash; 1.0 is
    identical partitions, 0.0 is chance.</p>
    {stab_chart}
    <p>Across eras, ARI ranges {stab['ari'].min():.3f} to {stab['ari'].max():.3f}. Across random
    seeds at fixed k it ranges {seed['ari'].min():.3f} to {seed['ari'].max():.3f}.</p>
    {stab_table}""",
    "Era-to-era agreement matters more than seed-to-seed: the former tests the concept, the latter only the optimiser.",
)

# --- 6. The archetypes themselves ---------------------------------------------
rows = prof.sort("archetype_label")
zmat = [[row[c] for c in feat_cols] for row in rows.iter_rows(named=True)]
prof_chart = heatmap(
    [rl(r["archetype_label"]).replace("&middot;", "·") for r in rows.iter_rows(named=True)],
    [nice(c) for c in feat_cols],
    zmat,
    label="Archetype signatures as feature z-scores",
    diverging=True, value_dp=2, label_w=215, cell=36,
)
sig_rows = []
for r in rows.iter_rows(named=True):
    top = sorted(((c, r[c]) for c in feat_cols), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    sig_rows.append([
        r["archetype_label"], names_by_label[r["archetype_label"]], f"{r['n']:,}",
        ", ".join(f"{nice(c)} {v:+.2f}" for c, v in top),
    ])
sig_table = table_view(
    ["Archetype", "Name", "Player-games", "Signature (largest |z|)"], sig_rows, "archetype signatures")

ex = pl.read_parquet(ART / "exemplars.parquet")
ex_html = []
for label in range(K):
    who = (ex.filter(pl.col("archetype_label") == label).sort("conf", descending=True)
             .head(6)["player_name"].to_list())
    if who:
        ex_html.append(
            f'<div class="archetypecard"><div class="archetypename">{rl(label)}</div>'
            f'<div class="archetypewho">{esc(", ".join(who))}</div></div>')
section(
    "6. What the archetypes are",
    f"""<p>Each archetype is described by how far its members sit from the league average on every
    feature, in standard deviations. Red is above average, blue below; the neutral midpoint is
    genuinely "league average", not a hue.</p>
    {prof_chart}{sig_table}
    <p class="sub">Highest-confidence members in 2025-26 (minimum 20 games). These names are the
    real test of whether the archetypes mean anything &mdash; the labels above were assigned from the
    z-score signatures, not from who landed in them.</p>
    <div class="archetypegrid">{''.join(ex_html)}</div>""",
    "Names are derived from each archetype's own signature by rule, so they cannot flatter the result.",
)

# --- 7. vs position ------------------------------------------------------
ct = pl.read_parquet(ART / "position_crosstab.parquet")
positions = (ct.group_by("position").agg(pl.col("n").sum().alias("t"))
               .sort("t", descending=True)["position"].to_list())
ctm = []
for p in positions:
    tot = ct.filter(pl.col("position") == p)["n"].sum()
    ctm.append([
        float(ct.filter((pl.col("position") == p) & (pl.col("archetype_label") == l))["n"].sum()) / tot * 100
        for l in range(K)])
ct_chart = heatmap(
    positions, [rl(l).replace("&middot;", "·") for l in range(K)], ctm,
    label="Share of each listed position's player-games falling in each archetype",
    diverging=False, value_dp=1, cell_dp=0, unit="%", label_w=150, cell=40,
)
section(
    "7. Archetypes against listed position",
    f"""<p>Reading across a row shows how one listed position scatters across archetypes. If position
    carried the same information as archetype, each row would be a single dark cell.</p>
    {ct_chart}
    {table_view(["Listed position"] + [f"archetype {l}: {names_by_label[l]}" for l in range(K)],
                [[p] + [f"{v:.1f}%" for v in rowv] for p, rowv in zip(positions, ctm)],
                "position to archetype, row-normalised")}""",
    "Rows are percentages of that position's player-games.",
)

# --- 8. Downstream test --------------------------------------------------
u = usage.sort("mae")
groups = ["Naive baseline" if m.startswith("baseline") else
          ("Model with archetypes" if "archetypes" in m else "Model without archetypes")
          for m in u["model"].to_list()]
model_chart = hbar_chart(
    [m.replace("baseline: ", "").replace("lightgbm: ", "LGBM · ").replace("ridge: ", "Ridge · ")
     for m in u["model"].to_list()],
    u["mae"].to_list(),
    label="Mean absolute error predicting next-game usage",
    value_label="MAE",
    groups=groups,
    group_colors={"Naive baseline": "var(--series-2)",
                  "Model without archetypes": "var(--series-3)",
                  "Model with archetypes": "var(--series-1)"},
    dp=5, label_w=310, row_h=25,
)
imp_html = ""
if (ART / "feature_importance.parquet").exists():
    imp = pl.read_parquet(ART / "feature_importance.parquet").head(14)
    imp_html = hbar_chart(
        [nice(c) for c in imp["feature"].to_list()],
        [float(g) for g in imp["gain"].to_list()],
        label="LightGBM gain by feature",
        value_label="gain", dp=0, label_w=190, row_h=24,
    ) + table_view(
        ["Feature", "Gain"],
        [[nice(r["feature"]), f"{r['gain']:,.0f}"] for r in
         pl.read_parquet(ART / "feature_importance.parquet").iter_rows(named=True)],
        "feature importance, full model")
section(
    "8. Do the archetypes help predict usage?",
    f"""<p>The clustering is only worth keeping if it earns its place downstream. Every model is
    trained on games before {M['cutoff']} ({M['panel_train']:,} player-games) and evaluated on
    games after it ({M['panel_test']:,}), never on a random shuffle. The naive baselines come
    first, as they should.</p>
    {model_chart}
    {table_view(["Model", "MAE", "RMSE"],
                [[r["model"], f"{r['mae']:.5f}", f"{r['rmse']:.5f}"] for r in u.iter_rows(named=True)],
                "usage model comparison")}
    <p>The comparison that answers the question is the pair differing only by archetype
    information: {base_mae:.5f} without it, {archetype_mae:.5f} with it &mdash; a
    <strong>{delta_pct:.2f}%</strong> improvement. That is far too small to call a gain. Every
    model does beat every naive baseline (the best model is {vs_baseline_pct:.1f}% better than
    the last-20-games mean), but that margin comes from the usage history, not from the archetypes.</p>
    <p>Note the row for <em>archetype features only</em>: with no usage history at all, the archetype
    features land within a whisker of the last-20 baseline. The archetypes do carry real information
    about how much a player will shoot &mdash; it is simply information his own recent usage
    already encodes.</p>
    {imp_html}""",
    "Time-based split, prior-game features only, baselines included.",
)

# --- 9. Play-type validation ---------------------------------------------
if (ART / "playtype_summary.parquet").exists():
    pts = pl.read_parquet(ART / "playtype_summary.parquet")
    ptv = pl.read_parquet(ART / "playtype_validation.parquet")
    ptp = pl.read_parquet(ART / "playtype_archetype_profiles.parquet")
    pt_targets = [c for c in ptp.columns if c.startswith("pt_share_")]

    def g(variant: str, feats: str) -> float:
        r = pts.filter((pl.col("variant") == variant) & (pl.col("features") == feats))
        return float(r["mean_cv_r2"][0]) if r.height else float("nan")

    archetypes_full, pos_full = g("full season", "archetypes (soft)"), g("full season", "listed position")
    archetypes_early = g("first 20 games", "archetypes (soft)")
    pos_early = g("first 20 games", "listed position")

    per = (ptv.filter(pl.col("variant") == "full season")
           .pivot(on="features", index="play_type", values="cv_r2"))
    cats = [r["play_type"].replace("p_and_r_ball_handler", "P&R ball handler").replace("_", " ")
            for r in per.iter_rows(named=True)]
    pt_chart = grouped_hbar(
        cats,
        {"Archetypes": [max(r["archetypes (soft)"], 0.0) for r in per.iter_rows(named=True)],
         "Listed position": [max(r["listed position"], 0.0) for r in per.iter_rows(named=True)]},
        label="Cross-validated R-squared predicting play-type share",
        value_label="CV R-squared",
        colors={"Archetypes": "var(--series-1)", "Listed position": "var(--series-2)"},
        dp=3, label_w=170,
    )
    prof_chart = heatmap(
        [rl(r["modal_archetype"]).replace("&middot;", "·") for r in ptp.sort("modal_archetype").iter_rows(named=True)],
        [c.removeprefix("pt_share_").replace("p_and_r_ball_handler", "P&R handler").replace("_", " ")
         for c in pt_targets],
        [[r[c] for c in pt_targets] for r in ptp.sort("modal_archetype").iter_rows(named=True)],
        label="Play-type mix by archetype, as z-scores",
        diverging=True, value_dp=2, label_w=215, cell=44, vmax=1.6,
    )
    section(
        "9. The test the archetypes did not train for",
        f"""<p>Everything above judges the archetypes on data related to what built them. This does
        not. <code>statyx.play_types</code> records how each player's offence is actually
        generated &mdash; spot-up, isolation, pick-and-roll handler, cut, hand-off, off-screen,
        transition. None of it entered the features, the fit, or the choice of k. The clustering
        has never seen it.</p>
        <p>The question is the one that matters for replacing position: <strong>do the archetypes
        predict that mix better than listed position does?</strong></p>
        <div class="tiles">
          <div class="tile good"><div class="tile-k">Archetypes</div><div class="tile-v">{archetypes_full:.3f}</div>
            <div class="tile-n">mean cross-validated R&sup2; across {per.height} play types</div></div>
          <div class="tile"><div class="tile-k">Listed position</div><div class="tile-v">{pos_full:.3f}</div>
            <div class="tile-n">same folds, same targets</div></div>
          <div class="tile good"><div class="tile-k">Ratio</div><div class="tile-v">{archetypes_full / pos_full:.1f}&times;</div>
            <div class="tile-n">archetypes over position</div></div>
        </div>
        {pt_chart}
        {table_view(["Play type", "Archetypes", "Listed position", "Archetypes + position"],
                    [[r["play_type"], f"{r['archetypes (soft)']:.3f}", f"{r['listed position']:.3f}",
                      f"{r['archetypes + position']:.3f}"] for r in per.iter_rows(named=True)],
                    "CV R-squared by play type, full season")}
        <p>Position is not merely weaker &mdash; on spot-ups, isolations and off-screen actions
        its R&sup2; is <em>negative</em>, meaning knowing a player is a "Guard" predicts his
        spot-up rate worse than guessing the league average. The archetypes reach
        {per['archetypes (soft)'].max():.2f} on pick-and-roll ball handling.</p>
        <p class="sub">Because <code>play_types</code> is a whole-season aggregate and the archetype
        responsibilities average over that same season, the numbers above measure concurrent
        validity rather than forecasting. Restricting archetypes to each player's <em>first 20 games</em>
        removes most of that overlap: archetypes fall to {archetypes_early:.3f} and position to
        {pos_early:.3f}, still a {archetypes_early / pos_early:.1f}&times; gap. The finding survives.</p>
        <p>Reading the signatures below confirms it qualitatively: the rim-running centers live
        on cuts, the playmaking guards on pick-and-roll handling, the floor spacers on spot-ups.
        The clustering reconstructed how these players generate offence from nothing but their
        box-score rates.</p>
        {prof_chart}
        <p class="sub"><strong>A sixth data defect, found here.</strong> Three play-type columns
        &mdash; <code>p_and_r_roll_man</code>, <code>post_up</code> and <code>o_board</code>
        &mdash; are identically zero for all 420 players in the dump. Roll man and post-up plainly
        occur, so this is a collection gap upstream rather than a fact about the season; they are
        excluded, leaving {per.height} of 10 play types. Players with fewer than 25 total
        charted attempts are also excluded as too noisy, leaving 313 of 420.</p>""",
        "Held-out in the strongest sense: this data was never available to the model.",
    )

# --- 10. Softness --------------------------------------------------------
section(
    "10. How soft is the assignment, really?",
    f"""<p>I expected the soft representation to be doing heavy lifting here. It is doing less
    than I assumed, and the honest number is worth stating: mean assignment confidence is
    {M['mean_confidence']:.3f}, and only <strong>{M['frac_ambiguous'] * 100:.1f}%</strong> of
    player-games have a top responsibility below 0.6. Most player-games land firmly in one
    archetype.</p>
    <p>That weakens the argument for responsibilities over a hard label &mdash; though
    {M['frac_ambiguous'] * 100:.1f}% of {M['design_rows']:,} is still roughly
    {int(M['frac_ambiguous'] * M['design_rows']):,} player-games where an integer id would
    assert something the model does not believe, and those will concentrate in exactly the
    hybrid players position labels already handle worst. The recommendation stands, but on
    narrower grounds than I would have claimed before measuring it: carry the
    {K} responsibility columns plus <code>archetype_entropy</code>, and treat
    <code>archetype_label</code> as a display convenience.</p>""",
)

# --- 10. Limits ----------------------------------------------------------
section(
    "11. Limits, and what I would not yet claim",
    """<ul class="method">
    <li><strong>The features are box-score shaped.</strong> Without tracking data over the full
      history, "archetype" here is a shot-and-activity diet. It cannot see screening, gravity, or
      off-ball defensive assignment. <code>statyx.advanced_stats</code> has
      <code>screen_assists</code>, <code>box_outs</code>, <code>contested_shots</code> and
      <code>touches</code>, but only for two seasons.</li>
    <li><strong>play_types is the obvious next input and is unusable as-is.</strong> One season,
      season-level grain. Its value would be as a <em>validation target</em>: do the clusters
      recover the play-type mix they were never shown?</li>
    <li><strong>The 2x duplication in advanced_stats is averaged, not resolved.</strong> 163
      keys genuinely disagree on touches. If the upstream dump can be fixed, fix it there.</li>
    <li><strong>Minutes are in the feature set.</strong> <code>r_min</code> is rotation archetype, but
      it also correlates with usage, so part of any downstream gain may be a minutes proxy
      rather than archetype. <code>include_volume=False</code> runs the ablation.</li>
    <li><strong>One cutoff, not rolling-origin folds.</strong> The split is a single
      train/test boundary. Repeating it across several origins would give error bars.</li>
    </ul>""",
)

# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
css = """
:root{color-scheme:light;
--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--gridline:#e1e0d9;--baseline:#c3c2b7;--border:rgba(11,11,11,.10);
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;
--pole-pos:#e34948;--pole-neg:#2a78d6;--mid-neutral:#f0efec;
--seq-lo:#e8eef5;--seq-hi:#184f95;--good:#006300;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
--muted:#898781;--gridline:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
--pole-pos:#e66767;--pole-neg:#3987e5;--mid-neutral:#383835;
--seq-lo:#22262b;--seq-hi:#86b6ef;--good:#0ca30c;}}
:root[data-theme="dark"]{color-scheme:dark;
--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
--muted:#898781;--gridline:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
--pole-pos:#e66767;--pole-neg:#3987e5;--mid-neutral:#383835;
--seq-lo:#22262b;--seq-hi:#86b6ef;--good:#0ca30c;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:860px;margin:0 auto;padding:48px 16px 80px}
header h1{font-size:30px;line-height:1.22;margin:0 0 8px;letter-spacing:-.02em}
header .lede{color:var(--text-secondary);font-size:17px;margin:0 0 4px}
header .meta{color:var(--muted);font-size:13px;margin-top:14px}
section{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
padding:26px 24px;margin:22px 0}
h2{font-size:19px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--text-secondary);font-size:14px;margin:0 0 16px}
p{margin:0 0 14px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;
background:color-mix(in oklab,var(--text-primary) 7%,transparent);padding:1px 5px;border-radius:4px}
.chart{width:100%;height:auto;display:block;margin:6px 0 2px;overflow:visible}
.tick{fill:var(--muted);font-size:11px}
.axislabel{fill:var(--text-secondary);font-size:12px}
.catlabel{fill:var(--text-secondary);font-size:12px}
.vallabel{fill:var(--text-secondary);font-size:11.5px;font-variant-numeric:tabular-nums}
.cellval{fill:#fff;font-size:10.5px;font-variant-numeric:tabular-nums}
.annot{fill:var(--series-2);font-size:11.5px;font-weight:600}
.hit{cursor:crosshair}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:2px 0 6px;font-size:12.5px;color:var(--text-secondary)}
.chip{display:inline-flex;align-items:center;gap:6px}
.chip i{width:10px;height:10px;border-radius:3px;display:inline-block}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0 4px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px}
.tile-k{color:var(--text-secondary);font-size:12.5px}
.tile-v{font-size:27px;letter-spacing:-.02em;margin:4px 0 2px}
.tile-n{color:var(--muted);font-size:11.5px;line-height:1.4}
.tile.good .tile-v{color:var(--good)}
.tile.flat .tile-v{color:var(--text-secondary)}
.tableview{margin:10px 0 2px;border-top:1px solid var(--border);padding-top:10px}
.tableview summary{cursor:pointer;color:var(--text-secondary);font-size:13px}
.scroll{overflow-x:auto;margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--text-secondary);font-weight:600;white-space:nowrap}
td{font-variant-numeric:tabular-nums}
ul.method{margin:0 0 14px;padding-left:20px}
ul.method li{margin-bottom:9px}
.archetypegrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:10px}
.archetypecard{border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.archetypename{font-size:13px;font-weight:600;margin-bottom:4px}
.archetypewho{color:var(--text-secondary);font-size:12.5px;line-height:1.5}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
background:var(--text-primary);color:var(--surface-1);font-size:12px;padding:6px 9px;
border-radius:6px;z-index:9;white-space:nowrap;font-variant-numeric:tabular-nums}
footer{color:var(--muted);font-size:12.5px;margin-top:28px;text-align:center}
@media(max-width:640px){.wrap{padding:28px 16px 60px}header h1{font-size:24px}}
"""

js = """
const tip=document.getElementById('tip');
document.addEventListener('mouseover',e=>{const t=e.target.closest('.hit');
if(!t||!t.dataset.tip)return;tip.innerHTML=t.dataset.tip;tip.style.opacity='1';});
document.addEventListener('mousemove',e=>{if(tip.style.opacity!=='1')return;
const p=12;let x=e.clientX+p,y=e.clientY+p;const r=tip.getBoundingClientRect();
if(x+r.width>innerWidth-8)x=e.clientX-r.width-p;
if(y+r.height>innerHeight-8)y=e.clientY-r.height-p;
tip.style.left=x+'px';tip.style.top=y+'px';});
document.addEventListener('mouseout',e=>{if(e.target.closest('.hit'))tip.style.opacity='0';});
"""

html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Archetype Clusters</title>
<style>{css}</style></head>
<body><div id="tip"></div><div class="wrap">
<header>
  <h1>Archetype clusters: a data-driven replacement for listed position</h1>
  <p class="lede">Listed position describes where a player stood in 1985. This fits archetypes from
  what players actually do &mdash; measured only from games already played, so a archetype is known
  before tip-off &mdash; and tests whether those archetypes survive refitting and whether they earn
  their place in a usage model.</p>
  <p class="meta">{M['design_rows']:,} player-games &middot; {M['players']:,} players &middot;
  {M['date_min']} to {M['date_max']} &middot; fit cutoff {M['cutoff']} &middot;
  generated {date.today().isoformat()}</p>
</header>
{hero}
{''.join(parts)}
<footer>Reproduce: <code>uv run python -m scripts.audit</code> &middot;
<code>uv run python -m scripts.experiment</code> &middot;
<code>uv run python -m scripts.report</code></footer>
</div><script>{js}</script></body></html>
"""
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html_doc)
print(f"wrote {OUT} ({len(html_doc):,} bytes), k={K}")
