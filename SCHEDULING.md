# Scheduling the player-archetypes job

Status: the job is scheduled through `nba/mod/_runner`.

## Schedule row

`util.update_schedule` carries one model row:

```text
job_family          model
schema              mod
table_name          player_archetypes
associated_function scs.nba.mod.player_archetypes
cadence             weekly
schema_sequence     NULL
function_sequence   NULL
pause               FALSE
```

For `job_family = 'model'`, `associated_function` is the repo identifier. The
runner derives the repo directory from the last dotted segment
(`player_archetypes`) and runs its root `__main__.py`.

## Execution

On the NUC:

```cron
0 14 * * 1 /home/oli/shaggy_camel_sports/nba/mod/_runner/cron.sh weekly \
  >> /home/oli/shaggy_camel_sports/nba/mod/_runner/cron.log 2>&1
```

Monday 14:00 UTC is after the database daily update at 13:00.

`_runner/cron.sh` runs `python __main__.py weekly`, which:

1. reads `job_family = 'model'` rows through `scs_hub`;
2. SSHes to the Jetson;
3. runs, from `/home/oli/shaggy_camel_sports/nba/mod/player_archetypes`:
   ```bash
   uv run --frozen --group data python __main__.py
   ```
4. writes `process start`, `player_archetypes.job`, and `process end` to
   `util.update_log` through `scs_runner`.

The database runner ignores the row because it selects
`job_family = 'database'`.

## Entry point and status

The root `__main__.py` calls `scripts.job.main()`. `scripts.job` still runs the
live pipeline and publisher sequentially, but when `SCS_RESULT_FILE` is set it
writes the row count there instead of writing its own `util.update_log` row.
Standalone runs without `SCS_RESULT_FILE` keep the old status-row behavior.

## Verification

From the NUC:

```bash
cd /home/oli/shaggy_camel_sports/nba/mod/_runner
./cron.sh --list
./cron.sh weekly --dry-run
./cron.sh --task player_archetypes
```

Then check the latest `util.update_log` rows for
`source = 'scs.nba.mod.player_archetypes'`: one `process start`, one
`player_archetypes.job`, one `process end`, with no duplicate
`player_archetypes.job` row.
