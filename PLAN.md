# Player Archetypes — Production Checklist

Status: 2026-11-19. Style-only taxonomy, one-table contract and intraday
assignment are live on the Jetson.

## Done

- **Style, not role.** The clustering uses 12 style rates. `start_rate` and
  `min` are `ROLE_FEATURES`, off by default and available only through
  `include_role=True` for the ablation.
- **Coverage fix.** A null `start_position` now means a bench game (0 starts),
  not a missing value. Zero-attempt games get zero shot shares, and a null
  trailing rate falls back to the player's last computed value.
- **Frozen fit, daily assign.** `model.pkl` fitted through 2026-10-20 at k=10.
  Each refit is archived under `artifacts/pipeline/models/<version>/` with a
  checksum; every output row carries `model_version`, `model_cutoff` and
  `model_checksum`.
- **k sweep.** Held-out log-likelihood and cluster share on live data: k=10 is
  the stopping point; k=12/14 gain 0.03–0.07 log-likelihood while the minimum
  cluster share halves.
- **External validation.** Mean cross-validated R² against `statyx.play_types`
  is 0.244 versus 0.067 for listed position; first-20-games 0.209 versus 0.073.
  The previous 14-feature model scored 0.174 and 0.130.
- **DB contract reset.** One private assignment table
  `mod.player_archetype_base`, one public view `mod.player_archetype_vw`, and a
  description lookup `mod.player_archetype_lookup_vw` keyed by
  `(model_version, archetype_label)`. The old current/profile/model tables and
  views are gone; `game_day_vw` reads the latest base row per player.
- **Coverage.** 4,417 current-season rows / 448 players through 2026-11-17.
  Cissoko runs through 11/17 (previously stopped at 11/02).
- **Schedule.** `util.update_schedule` is `intraday`, function sequence 0,
  before `team_hierarchy`.
- **Data access.** Production and research read the public views through
  `scs-hub`; the Parquet snapshot and its `gen.py` are gone.

## Open

- Players with no 2026-27 appearance (82 in `game_day_vw`) have no row, because
  only the current season is published. A last-known carry-forward row is the
  intended fix, and it needs an explicit decision on whether those rows carry
  their true season or a synthetic date.
- Confidence is still a mixture posterior. Reliability has not been measured;
  entropy is the safe uncertainty measure until it is.
- Era-to-era stability (ARI) has not been re-measured on the style-only
  taxonomy.
- Absorption similarity was ablated and rejected with season-aware walk-forward
  vectors: 4.650 versus 4.652 minutes MAE overall and 5.117 versus 5.131 on the
  top quartile; the starter-weighted variant was 4.652. Revisit only with a
  materially better taxonomy or player coverage.
- CI for ruff and pytest is still absent.

## Definition of done

- The intraday job runs unattended on the Jetson and fails visibly on stale
  data, a missing model or a failed publish.
- Every published row is reproducible from the archived model version.
- The tables contain exactly the current regular season plus the agreed
  carry-forward rows.
