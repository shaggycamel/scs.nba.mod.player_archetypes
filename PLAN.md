# Player Archetypes — Remaining Work

Status: 2026-11-02. The model and the data stage run end to end; nothing is
scheduled yet.

## Done

- `scripts/pipeline.py` loads or fits the mixture, assigns roles, and publishes
  `assignments.parquet`, `role_profiles.parquet` and `run.json`.
- `scripts/refresh_data.py` pulls the seven model inputs through `scs-hub` using
  the read-only `[cockroach-read]` connection, stages them, and writes
  `_snapshot.json`.
- Pipeline guards: `--require-snapshot` and `--max-snapshot-age-days`.
- Optional `data` dependency group; `uv sync --group data`.
- Live snapshot verified: last played 2026-10-31, schedule through 2027-04-11.
- Java 17, `libpq`, and credentials on the Jetson are working.
- Ruff clean; 10 tests pass.

## P0 — Make the scheduled job real

### 1. Choose the scheduler and cadence

Recommended:

- Nightly at 09:00 UTC: refresh data, then assign with the saved model.
- Weekly (Monday) after the nightly refresh: refit, then assign.
- systemd timer on the Jetson, not cron: journald logs, `Persistent=true`, and a
  visible failed state.

Open decision: systemd vs cron; exact times; whether the weekly refit is Monday
or another day.

### 2. Do the first production refit

The saved `artifacts/pipeline/model.pkl` was fitted with cutoff `2024-10-01`
(the research cutoff), so it does not describe 2024–2026 role drift. A
production model should be fitted on all games before the run date:

```bash
uv run python -m scripts.pipeline --refit --k 10 --cutoff "$(date -I)"
```

Decide the refit cutoff policy. Recommended: weekly refit with cutoff = run
date, k fixed at 10. Re-run the k sweep only when the data volume or feature set
changes materially.

### 3. Add the current-player output

The pipeline currently publishes historical player-games. Add
`current_roles.parquet`: one row per player from their most recent assignment,
carrying team, game date, responsibilities, label, confidence and entropy.
Decide whether “current” means the latest game or the latest N-game average.

### 4. Wire the two-stage job

Create the systemd units (or cron entries) and a small wrapper:

```bash
# refresh
uv run --group data python -m scripts.refresh_data
# assign
uv run python -m scripts.pipeline --require-snapshot --max-snapshot-age-days 7
```

Requirements:

- run from the repo root;
- `PLAYER_ARCHETYPES_DATA` points at the snapshot directory;
- `SCS_HUB_CREDENTIALS` is set only if the credentials file is not at
  `~/.config/scs_hub_credentials.ini`;
- the assign step does not run if the refresh step fails;
- logs go to journald;
- a failed daily run is visible (systemd `OnFailure=` or an alert hook).

### 5. Commit and put it on a branch

Everything is currently uncommitted on `main`. Create a branch, commit the
pipeline, refresh script, tests, `pyproject.toml`, `uv.lock` and README, and
push. Decide whether to open a pull request.

## P1 — Robustness

### 6. Snapshot validation

`_snapshot.json` records row counts and the last played date, but nothing checks
them yet. Add:

- expected tables and columns present;
- no empty tables;
- row counts within a tolerance of the previous snapshot;
- played-game date does not move backwards;
- fail the refresh rather than replacing a good snapshot with a bad one.

### 7. Versioned runs and model metadata

- Write each run to `artifacts/runs/<timestamp>/` and maintain a `latest`
  pointer, with retention.
- Persist model metadata with the model itself: feature list, window,
  min-periods, k, seed, fit cutoff, fit rows, package versions, git SHA.
- Keep `run.json` as the human-readable summary.

### 8. Structural monitoring

The model has no accuracy score, so monitor structure instead:

- cluster shares;
- mean confidence and fraction below 0.6;
- ARI between the new fit and the previous fit;
- feature-distribution drift.

Flag a refit that collapses a cluster or changes the partition sharply. This is
the replacement for the research MAE check, which does not belong in the job.

### 9. Report regeneration

`scripts/report.py` expects the research artifacts (k sweep, stability,
ablation). Decide whether the scheduled job should:

- keep the research report frozen and publish only tables, or
- add a lightweight production report driven by `run.json` and the profiles.

## P2 — Later

- Automate the k sweep on a slow cadence and review it rather than refitting k
  every week.
- Backfill support: refit and assign a historical date range on demand.
- CI for ruff and pytest.
- Runbook: how to refresh credentials, read the logs, recover from a failed
  snapshot, and roll back a bad model.
- Decide whether `scripts/experiment.py` remains the only place that reproduces
  the published research numbers.

## Definition of done

- The timer runs unattended on the Jetson.
- A fresh snapshot and a fresh assignment land every day.
- `current_roles.parquet` answers “what role is each player in now?”.
- A failed refresh or stale snapshot stops the assign step and is visible.
- Each run records enough metadata to reproduce it.
- The code is committed on a branch and pushed.
