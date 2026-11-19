# player-archetypes

Point-in-time player archetypes for basketball player-games.

This package replaces listed position with a soft, data-driven description of
what a player actually does on the floor. A Gaussian mixture is fitted on 12
trailing style rates, then scores each player-game with a vector of archetype
responsibilities. Role and volume are deliberately excluded; they belong in the
minutes and absorption models.

Extracted from the `nba_adhoc` role-clustering branch at commit
`2d93ab24755bc50a728494916f90d1ef91000e75` (`cluster-mod`). The source branch
is archived in that repository as tag `archive/cluster-mod-2026-11-02`.

## What it produces

- `archetype_0` … `archetype_{k-1}` — soft responsibilities, one per archetype; rows sum to 1.
- `archetype_label` — the argmax archetype, for reading and plotting.
- `archetype_confidence` — the largest responsibility. It is a mixture
  posterior, not a calibrated probability; prefer `archetype_entropy` when the
  uncertainty matters.
- `archetype_entropy` — normalised entropy; 0 is pure, 1 is evenly split.
- `features_through` — the last game that fed the assignment, so staleness is
  visible without guessing from the run date.

The labels are convenience output. The responsibilities are the representation:
a player who does two jobs gets weight on two archetypes instead of being forced into
one integer category.

The production surface is one assignment table, `mod.player_archetype_base`,
one public view, `mod.player_archetype_vw`, and a small description lookup,
`mod.player_archetype_lookup_vw`. The lookup has one row per
`(model_version, archetype_label)`; the assignment view joins it so every row
carries `archetype_description`. Feature z-scores still live in
`artifacts/pipeline/run.json`.

## Design rules

- **Style, not role or volume.** Counting stats are per-minute and shot types
  are shares of the player's own attempts. `start_rate` and `min` are role
  features, kept in `ROLE_FEATURES` and off by default.
- **No usage.** `usg_pct` and `e_usg_pct` are the downstream target and never
  enter the clustering inputs.
- **No efficiency.** Shooting percentages describe how well, not what.
- **No team context.** Ratings, pace and possession counts describe the five on
  the floor.
- **Strictly prior games.** Every feature at game *g* uses games before *g*, and
  every fit is handed only rows before a cutoff date.
- **Last-known fallback.** A null trailing style rate is carried forward from
  the player's previous computed value, so a short-handed window does not drop
  the row.

## Results

The style-only taxonomy is a better description of how players play than listed
position. Against `statyx.play_types` — data the clustering never saw — the
archetypes reach mean cross-validated R² **0.244 versus 0.067** for listed
position (3.6×). The early-season variant, which removes most same-season
overlap, gets **0.209 versus 0.073** (2.9×). The previous 14-feature model
scored 0.174 and 0.130 on the same tests; removing role and fixing the
`start_rate` null improved both.

The usage result from the previous iteration still stands: adding archetype
features to a next-game usage model moved MAE by 0.25% and was rejected. It has
not been retested on the new taxonomy, and the open questions below remain.
Era-to-era stability and confidence calibration are the next measurements, not
claims.

The full previous-iteration analysis is in
[`docs/archetype_clustering_report.html`](docs/archetype_clustering_report.html).

## Install

```bash
uv sync --group dev
```

The core model needs only `polars`, `numpy` and `scikit-learn`; the `data` group
adds `scs-hub` for the live views. The research harness also uses `pyarrow` for
its analysis artifacts, and the optional `experiments` extra adds LightGBM for
the full downstream comparison.

## Data

The database is the only source. The pipeline and the research loaders read the
public views through `scs-hub`; there is no Parquet snapshot and no
`gen.py`.

```bash
uv sync --group data
uv run --group data python -m scripts.pipeline
```

The default connection is `jetson-read`; override it with
`PLAYER_ARCHETYPES_DB_CON` (or the `db_con` argument on the loader). The
production job passes `--db-con jetson-write` because it runs on the Jetson and
publishes from the same role.

Credentials come from `SCS_HUB_CREDENTIALS` or
`~/.config/scs_hub_credentials.ini`. The host needs `libpq` to build
`psycopg2` and a headless JRE (`openjdk-17-jre-headless`) because `scs-hub`
imports `nbainjuries` at load time.

## Use

```python
from datetime import date

from player_archetypes import fit, features
from player_archetypes.data import load

box = load.box_scores()
pg = features.usage_history(features.per_game(box))
rolled = features.rolling(pg, window=20, min_periods=5)
design, names = features.design_matrix(rolled)

model = fit(design, names, k=10, cutoff=date(2024, 10, 1))
assigned = model.assign(design)
```

`fit` returns an `ArchetypeModel` carrying the scaler, optional PCA and mixture, so
`assign` can be applied to future rows without refitting. `RoleModel` remains as
an alias for models pickled before the rename.

## Scheduled pipeline

The scheduled entry point is the repository-root `__main__.py`, run by
`nba/mod/_runner` on the NUC:

```bash
uv sync --group data
uv run --frozen --group data python __main__.py
```

`__main__.py` calls `scripts.job`. It reads live, loads the frozen seasonal
model, assigns the current season, and publishes. When `SCS_RESULT_FILE` is set
(the runner path), it writes `{"row_count": n}` there and lets the runner own
`util.update_log`; standalone runs keep the old status-row behavior.

For a manual standalone run:

```bash
uv run --group data python -m scripts.job --cadence intraday
```

It fails before assigning if the newest played game is more than seven days old
(`--max-data-age-days`), if `model.pkl` is missing, or if the publish fails.
The pipeline and publish stages remain available separately for debugging.

The schedule row runs `intraday` at function sequence 0, before
`team_hierarchy`; see [`SCHEDULING.md`](SCHEDULING.md).

Refit only once per season:

```bash
uv run --group data python -m scripts.pipeline \
  --refit --cutoff 2026-10-20 --k 10 \
  --window 10 --min-periods 5 --min-prior-games 5 \
  --sample 0 --regular-season-only
```

Each refit is archived under `artifacts/pipeline/models/<model_version>/` with
`model.json` and a checksum; every published row carries `model_version`,
`model_cutoff` and `model_checksum`.

`scripts/publish.py` writes the private `mod.player_archetype_base` table
through the `jetson-write` connection. The public `mod.player_archetype_vw`
view is applied from `nba/database/sql/views/mod/`; `kobe_public` only gets
`SELECT` on the view. It publishes the current season only by default;
`--all-seasons` is the explicit retrospective backcast.

Both scripts exit non-zero on failure, so they can be called directly from cron
or a systemd timer.

## Reproduce

```bash
uv run --group data python -m scripts.experiment        # k sweep, stability, ablation, artifacts/
uv run --group data python -m scripts.validate_playtypes  # external play-type validation
uv run --group data python -m scripts.backfill_assignments  # walk-forward vectors for team_hierarchy
uv run python -m scripts.report                           # docs/archetype_clustering_report.html
```

The research loaders read the same public views as production, so there is no
snapshot to set up. `scripts/report.py` writes
`docs/archetype_clustering_report.html` from the analysis artifacts in
`artifacts/`.

## Layout

```
src/player_archetypes/
  cluster.py      # mixture fit, assignment, k selection, stability, profiling
  features.py     # per-game rates, trailing windows, usage baselines
  data/           # database loaders and id crosswalks
scripts/          # experiment, validation and report harness
tests/            # synthetic-data tests; no database required
docs/             # rendered report
```
