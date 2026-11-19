# Scheduling the player-archetypes job

Status: the job is scheduled through `nba/mod/_runner` on the intraday cadence,
before `team_hierarchy`.

## Schedule row

`util.update_schedule` carries one model row:

```text
job_family          model
schema              mod
table_name          player_archetypes
associated_function scs.nba.mod.player_archetypes
cadence             intraday
schema_sequence     NULL
function_sequence   0
pause               FALSE
```

`function_sequence = 0` places it ahead of `injury_daily_forecast` (1),
`team_hierarchy` (2) and `player_stat_forecast` (3), so `team_hierarchy` can
consume the current responsibilities. For `job_family = 'model'`,
`associated_function` is the repo identifier; the runner derives the directory
from the last dotted segment (`player_archetypes`) and runs the root
`__main__.py` on the Jetson.

## Execution

The model runner runs the intraday cadence at 16:15 and 22:15 UTC, after the
database intraday refresh at 16:00 and 22:00. On the NUC:

```cron
15 16 * * * /home/oli/shaggy_camel_sports/nba/mod/_runner/cron.sh intraday \
  >> /home/oli/shaggy_camel_sports/nba/mod/_runner/cron.log 2>&1
15 22 * * * /home/oli/shaggy_camel_sports/nba/mod/_runner/cron.sh intraday \
  >> /home/oli/shaggy_camel_sports/nba/mod/_runner/cron.log 2>&1
```

`_runner/cron.sh` runs `python __main__.py intraday`, which:

1. reads `job_family = 'model'` rows through `scs_hub`;
2. orders by `schema_sequence`, then `function_sequence`;
3. SSHes to the Jetson;
4. runs, from `/home/oli/shaggy_camel_sports/nba/mod/player_archetypes`:
   ```bash
   uv run --frozen --group data python __main__.py
   ```
5. writes `process start`, `player_archetypes.job`, and `process end` to
   `util.update_log` through `scs_runner`.

## Entry point and status

The root `__main__.py` calls `scripts.job.main()`. `scripts.job` runs the live
pipeline and publisher sequentially. When `SCS_RESULT_FILE` is set it writes the
base row count there and lets the runner own `util.update_log`; standalone runs
keep the old status-row behaviour.

The pipeline loads the frozen model and assigns the current season. It does not
refit. A refit is a separate, deliberate run:

```bash
uv run --group data python -m scripts.pipeline \
  --refit --cutoff 2026-10-20 --k 10 \
  --window 10 --min-periods 5 --min-prior-games 5 \
  --regular-season-only
```

## Verification

From the NUC:

```bash
cd /home/oli/shaggy_camel_sports/nba/mod/_runner
./cron.sh --list
./cron.sh intraday --dry-run
./cron.sh --task player_archetypes
```

Then check the latest `util.update_log` rows for
`source = 'scs.nba.mod.player_archetypes'`: one `process start`, one
`player_archetypes.job`, one `process end`, with no duplicate
`player_archetypes.job` row, and `row_count` equal to the current-season rows in
`mod.player_archetype_base`.
