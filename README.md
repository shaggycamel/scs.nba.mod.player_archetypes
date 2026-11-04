# player-archetypes

Point-in-time role archetypes for basketball player-games.

This package replaces listed position with a soft, data-driven description of
what a player actually does on the floor. A Gaussian mixture is fitted on
trailing, per-minute rate stats, then scores each player-game with a vector of
role responsibilities.

Extracted from the `nba_adhoc` role-clustering branch at commit
`2d93ab24755bc50a728494916f90d1ef91000e75` (`cluster-mod`). The source branch
is archived in that repository as tag `archive/cluster-mod-2026-11-02`.

## What it produces

- `role_0` … `role_{k-1}` — soft responsibilities, one per role; rows sum to 1.
- `role_label` — the argmax role, for reading and plotting.
- `role_confidence` — the largest responsibility.
- `role_entropy` — normalised entropy; 0 is pure, 1 is evenly split.

The labels are convenience output. The responsibilities are the representation:
a player who does two jobs gets weight on two roles instead of being forced into
one integer category.

## Design rules

- **Rate, not volume.** Counting stats are per-minute; shot types are shares of
  the player's own attempts.
- **No usage.** `usg_pct` and `e_usg_pct` are the downstream target and never
  enter the clustering inputs.
- **No efficiency.** Shooting percentages describe how well, not what.
- **No team context.** Ratings, pace and possession counts describe the five on
  the floor.
- **Strictly prior games.** Every feature at game *g* uses games before *g*, and
  every fit is handed only rows before a cutoff date.

## Results

The roles are a better description of how players play than listed position,
and they do not improve a next-game usage model. Both results are in
[`docs/role_clustering_report.html`](docs/role_clustering_report.html).

- Against `statyx.play_types` — data the clustering never saw — the roles reach
  mean cross-validated R² 0.174 versus 0.047 for listed position (3.7×). The
  early-season variant, which removes most same-season overlap, still gets
  0.130 versus 0.058 (2.2×).
- Adding role features to a usage model moves MAE from 0.05016 to 0.05004, a
  0.25% improvement. On the project's rule — keep clustering only if an
  ablation shows a gain — this model does not qualify.
- Era-to-era ARI is 0.467–0.666; seed-to-seed ARI is 0.656–0.996. The taxonomy
  is recognisable but its boundaries move with the fitting window.

The fair summary: a descriptive taxonomy, not a useful next-game usage feature.

## Install

```bash
uv sync --group dev
```

The core model needs only `polars`, `numpy` and `scikit-learn`; the parquet
loaders also need `pyarrow`. The optional `experiments` extra adds LightGBM for
the full downstream comparison.

## Data

The source snapshot is not committed. Point the loaders at a directory
containing `nba/`, `statyx/` and `util/`:

```bash
export PLAYER_ARCHETYPES_DATA=/path/to/nba_adhoc/data
```

Every loader also accepts an explicit `data=` argument. If the environment
variable is unset, the default is `data/` in the repository root.

The model needs these files in particular:

```
nba/player_box_score.parquet
nba/league_game_schedule.parquet
nba/player_info.parquet
statyx/advanced_stats.parquet
statyx/play_types.parquet
statyx/schedule.parquet
util/player_id_map_vw.parquet
```

### Read from the live database

The pipeline reads the live database by default through `scs-hub` and its
read-only `cockroach-read` connection:

```bash
uv sync --group data
uv run --group data python -m scripts.pipeline
```

`scripts/pipeline.py` selects only the box-score and schedule columns the
features use, so it does not depend on a local snapshot. The Parquet loaders
remain for the research scripts, but production has no offline path.

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

`fit` returns a `RoleModel` carrying the scaler, optional PCA and mixture, so
`assign` can be applied to future rows without refitting.

## Scheduled pipeline

The weekly production entry point is `scripts.job`. It reads live, loads the
frozen seasonal model, assigns the current season, publishes, and writes a
status row to `util.update_log`:

```bash
uv sync --group data
uv run --group data python -m scripts.job --cadence weekly
```

It fails before assigning if the newest played game is more than seven days old
(`--max-data-age-days`), if `model.pkl` is missing, or if the publish fails.
The pipeline and publish stages remain available separately for debugging.

Refit only once per season:

```bash
uv run --group data python -m scripts.pipeline \
  --refit --cutoff 2026-10-20 --k 10 \
  --window 10 --min-periods 5 --min-prior-games 5 \
  --sample 0 --regular-season-only
```

`scripts/publish.py` writes the `mod.player_role*` tables through the
`cockroach-write` connection. It publishes the current season only by default;
`--all-seasons` is the explicit retrospective backcast.

Both scripts exit non-zero on failure, so they can be called directly from cron
or a systemd timer.

## Reproduce

```bash
export PLAYER_ARCHETYPES_DATA=/path/to/nba_adhoc/data
uv run python -m scripts.experiment        # k sweep, stability, ablation, artifacts/
uv run python -m scripts.validate_playtypes  # external play-type validation
uv run python -m scripts.report              # docs/role_clustering_report.html
uv run python -m scripts.audit               # data defects
```

`scripts/report.py` writes `docs/role_clustering_report.html`. The committed
report is the output of the commands above against the source snapshot.

## Layout

```
src/player_archetypes/
  cluster.py      # mixture fit, assignment, k selection, stability, profiling
  features.py     # per-game rates, trailing windows, usage baselines
  data/           # parquet loaders and id crosswalks
scripts/          # experiment, validation, audit and report harness
tests/            # synthetic-data tests; no snapshot required
docs/             # rendered report
```
