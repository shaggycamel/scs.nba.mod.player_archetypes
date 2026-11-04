# Scheduling the player-archetypes job

Status: 2026-11-04. The model, live pipeline, publisher and run logging are
working. The remaining work is registering the job in `util.update_schedule`
and choosing where it executes.

## Current state

- `scripts/job.py` runs the live pipeline, then the publisher, and writes one
  `util.update_log` row per run.
- The pipeline fails on stale data, a missing model, or a failed publish.
- `mod.player_role` contains only the current regular season.
- The model is frozen for the season; do not refit weekly.

## What "scheduled in util.update_schedule" means

`util.update_schedule` is the database's schedule of record. Each row names an
`associated_function`, a `cadence`, and an ordering position. The database
runner (`scs.nba.database/__main__.py`) reads the table and calls each function.

A row alone is not enough. The runner resolves:

```python
hub.<associated_function>
```

against `SportsHub`. There is currently no `player_archetypes` attribute, so a
row pointing at `player_archetypes.job` would fail at runtime.

## Constraints

1. **Host split**
   - `util.update_schedule` is executed by the database runner on the NUC.
   - The model, `model.pkl` and this repository are on the Jetson.
   - The NUC cannot run a Jetson script unless the model is deployed there or a
     remote trigger is added.

2. **Weekly timing**
   - The database weekly cron runs Monday **12:00 UTC**.
   - The NBA daily update runs at **13:00 UTC**.
   - A `weekly` row would therefore miss Sunday night's games.
   - The job needs a weekly-after-daily slot around **14:00 UTC**, or a new
     cadence that is only invoked after the daily update.

3. **Runner bridge**
   - Either add a `player_archetypes` component to `scs-hub`, or extend the
     database runner to execute an external command.
   - An `associated_function` row without one of those bridges will not run.

## Option A — run on the NUC

Use this if the database host can own the model compute.

1. Deploy this repository and `artifacts/pipeline/model.pkl` to the NUC.
2. Install the model dependencies there (`uv sync --group data`).
3. Add one of:
   - a `player_archetypes` component/method to `scs-hub`, or
   - an external-command hook to `scs.nba.database/__main__.py`.
4. Add a new weekly-after-daily cadence, for example `weekly_after_daily`, at
   14:00 UTC in the NUC crontab.
5. Insert the schedule row:

```sql
INSERT INTO util.update_schedule
    (schema, table_name, associated_function, cadence,
     schema_sequence, function_sequence, pause)
VALUES
    ('mod', 'player_role', 'player_archetypes.run', 'weekly_after_daily',
     5, NULL, FALSE);
```

6. Apply the row to both Postgres and CockroachDB, following the database
   repo's normal schedule-row process.
7. Verify with a manual `./cron.sh weekly_after_daily` run and confirm a
   `util.update_log` row with `task = 'player_archetypes.run'`.

## Option B — run on the Jetson

Use this if the model and `model.pkl` must stay on the Jetson.

1. Add a model-specific cadence row that the NUC cron never selects, for
   example `mod_weekly`:

```sql
INSERT INTO util.update_schedule
    (schema, table_name, associated_function, cadence,
     schema_sequence, function_sequence, pause)
VALUES
    ('mod', 'player_role', 'player_archetypes.job', 'mod_weekly',
     5, NULL, FALSE);
```

2. Add a Jetson-side runner that reads `util.update_schedule` for
   `cadence = 'mod_weekly'` and `pause = FALSE`, then calls
   `scripts.job --cadence mod_weekly`.
3. Add the Jetson crontab entry for Monday 14:00 UTC. The cron only triggers
   the runner; the database row decides whether the job runs.
4. Ensure the NUC never invokes `mod_weekly`; its crontab only passes
   `daily`, `intraday` and `weekly`. A bare `./cron.sh` would pick up
   `mod_weekly` and fail, so avoid bare runs or make the NUC runner ignore
   model-specific cadences.
5. Verify the Jetson runner creates a `util.update_log` row and the current
   season rows in `mod.player_role`.

## Option C — NUC triggers the Jetson over SSH

This keeps the model and `model.pkl` on the Jetson while letting the NUC own
the schedule. It is the recommended option when the model must not move.

1. Create a dedicated SSH key on the NUC for the trigger.
2. On the Jetson, restrict that key in `~/.ssh/authorized_keys` to one forced
   command, for example:

   ```text
   command="/home/oli/shaggy_camel_sports/nba/mod/player_archetypes/scripts/remote_job.sh",no-agent-forwarding,no-port-forwarding,no-pty,no-user-rc,no-X11-forwarding ssh-ed25519 AAAA... nuc-archetypes
   ```

3. `scripts/remote_job.sh` runs:

   ```bash
   exec uv run --group data python -m scripts.job --cadence weekly_after_daily
   ```

4. Add the `util.update_schedule` row with cadence `weekly_after_daily`, as in
   Option A. The NUC cron passes that cadence at 14:00 UTC, after the daily
   NBA update.
5. Bridge the runner to SSH. Either:
   - add an external-command hook to `scs.nba.database/__main__.py` that maps
     `player_archetypes.trigger` to the SSH command, or
   - run a host-side NUC cron wrapper that reads the schedule row and then
     calls SSH.
6. The NUC container needs an SSH client and the private key if the bridge
   lives in the database runner. A host-side wrapper avoids mounting a key
   into the container.
7. Let the Jetson job write the detailed `util.update_log` row. The NUC
   trigger should log a separate `player_archetypes.trigger` row, or no row,
   so the same run is not double-counted.

Example SSH invocation:

```bash
ssh -i /path/to/nuc_archetypes_key \
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=30 \
  oli@jetson.tail195465.ts.net
```

## Decision required

Choose the compute host:

- **NUC** — the existing database scheduler owns execution; the model and
  `model.pkl` must be deployed there.
- **Jetson** — the model stays where it was trained; either a Jetson-side
  runner polls `util.update_schedule` (Option B), or the NUC triggers it over
  SSH (Option C). Option C is the cleaner fit for the existing scheduler.

## Definition of done

- Exactly one `util.update_schedule` row exists for the archetype job.
- The row runs once a week after the NBA daily update.
- A successful run writes both the `mod.player_role*` tables and a
  `util.update_log` row.
- A failed or stale run is visible in `util.update_log` and the cron log.
