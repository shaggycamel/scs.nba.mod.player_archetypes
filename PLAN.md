# Player Archetypes — Production Checklist

Status: 2026-11-04. The model, live pipeline and Cockroach output are working.
The scheduler and operational safeguards are what remain.

> Scheduling update: the weekly job now runs through `nba/mod/_runner` using
> `util.update_schedule.job_family = 'model'`. See `SCHEDULING.md`.

## Done

- Frozen production model: regular season only, 10-game rolling window,
  fit cutoff 2026-10-20, trained through the end of 2025-26.
- `scripts/pipeline.py` reads live from `[cockroach-read]`; no Parquet fallback.
- Regular-season filtering happens before rolling features.
- Model metadata (window, min-periods, prior-games, season type, fit rows, seed)
  is saved in `model.pkl` and checked on load.
- Missing `model.pkl` fails the run instead of silently refitting.
- `scripts/publish.py` creates and fills the `mod.player_archetype*_base` tables
  through `[cockroach-write]`.
- The public `mod.player_archetype*_vw` views are applied from
  `nba/database/sql/views/mod/`; read access is granted there.
- The publisher writes the current season only by default; historical backcasts
  require `--all-seasons`.
- Live output verified through 2026-11-02: 2,063 player-games, 390 current
  players, one model row.
- `scripts/job.py` runs pipeline then publish and writes one `util.update_log`
  row per run.
- Freshness guard: fails if the newest played game is more than seven days old.
- Atomic publish: the current-season replacement runs in one transaction.
- Pre-publish validation: no null keys, no duplicates, responsibilities sum to 1.
- Ruff clean; tests pass.

## P0 — before the weekly cron

1. **Cron wrapper and entry**
   - Call `uv run --frozen --group data python -m scripts.job --cadence weekly`.
   - `flock` to prevent overlap.
   - Explicit `PATH`; run from the repo root.
   - Monday 14:00 UTC, after the database's 13:00 UTC daily update.
   - Append to `cron.log`; exit non-zero on failure.

## P1 — robustness

5. **Model archive and versioning**
   - Archive each seasonal fit under a versioned path, not only the mutable
     `artifacts/pipeline/model.pkl`.
   - Record a checksum and the model ID in the output tables.

6. **Post-run validation**
   - Expected current-season row count.
   - No null keys or duplicate `(game_id, player_id)`.
   - Responsibility rows sum to 1.
   - Cluster shares within a sane range.
   - Mean confidence and ambiguous share.

7. **CI**
   - GitHub Actions for ruff and pytest.

8. **Runbook**
   - Seasonal refit command.
   - How to read logs and `util.update_log`.
   - How to recover from a failed publish.
   - How to roll back a bad model.

9. **DDL ownership**
   - Keep the `mod` table definitions in the repo as SQL or version them in the
     publisher, rather than relying on implicit `CREATE TABLE IF NOT EXISTS`.

## P2 — later

10. **Seasonal refit reminder**
    - Refit in preseason with cutoff at the first regular-season game.
    - Keep the taxonomy frozen for the season.

11. **Archetype stability monitoring**
    - Compare seasonal fits (ARI, cluster shares, confidence).
    - Flag a refit that collapses or reshuffles the taxonomy.

## Definition of done

- The weekly cron runs unattended on the Jetson.
- The job fails visibly on a stale database, missing model or failed publish.
- Every run leaves a status row and a reproducible model version.
- The `mod` tables always contain exactly the current regular season.
