# Player archetypes: next iteration

Status: 2026-11-19. Notes for the next pass on `scs.nba.mod.player_archetypes`.
This is a design handoff, not a production checklist.

Implementation status: the style/role split, `start_rate` null fix, last-known
fallback, archived fit/checksum, one-table contract, intraday assignment and the
play-type validation are done (mean R² 0.244 versus 0.067 for position). Still
open: carry-forward rows for players without a 2026-27 game, confidence
calibration, and ARI stability.

Absorption ablation (2026-11-19): walk-forward assignments for 2021-26,
Bhattacharyya similarity. Minutes MAE 4.649 versus 4.652, top-quartile vacated
minutes 5.127 versus 5.131, starter recall slightly worse. The feature was
rejected and production absorption keeps the position/start-rate feature set.

## Current state

- GMM, `k=10`, 10-game rolling window, `min_periods=5`, `min_prior_games=5`.
- Production features are 14: the 12 style rates, plus **`r_start_rate`** and
  **`r_min`**.
- Fitted through 2025-26 with cutoff 2026-10-20, frozen for the season, assigned
  by a **weekly** job.
- Outputs soft responsibilities, an argmax label, confidence and entropy.
  `mod.game_day_vw` joins the current label for display only.
- The live model row reports mean confidence 0.929 and 4.8% ambiguous rows.

## What works

- The soft responsibilities are the right representation; the argmax label is
  not.
- Features are point-in-time and exclude usage, efficiency and team context.
- The taxonomy beats listed position against `statyx.play_types` (mean
  cross-validated R² 0.174 versus 0.047).
- It is reasonably stable across eras (ARI 0.467-0.666).

## What is wrong

1. **The production model is not style-only.** `r_min` and `r_start_rate` are
   clustering inputs even though the README says "rate, not volume." Cluster 7,
   "Starting big," contains Scoot Henderson (PG), Toumani Camara (PF) and
   Donovan Clingan (C). It is a **starter cluster**, not a style cluster. Role
   and style are conflated.
2. **Assignments are stale and incomplete.** The weekly job left Sidy Cissoko
   on his 11/2 assignment ("Deep reserve," confidence 0.9998) after he had
   started five games. `mod.player_archetype_vw` has only six Sidy rows through
   11/2, while other POR players run through 11/13. The design matrix drops rows
   with any null feature, so coverage is not guaranteed. Diagnose the null
   feature and add imputation or a last-known fallback.
3. **No predictive integration.** The archetype is joined into
   `mod.game_day_vw` for display only. It is not in the minutes, absorption,
   usage or player-stat features.
4. **The only predictive test was usage**, which was rejected at +0.25%. That
   does not settle minutes or absorption, but it means do not retest the same
   usage formulations.
5. **Confidence is overconfident.** The GMM gives near-1.0 confidence for
   ordinary role assignments. Calibrate with temperature scaling or report
   entropy rather than treating confidence as a probability without validation.
6. **Coverage and versioning are weak.** The model artifact is mutable; there
   is no archived seasonal fit or checksum; the tables do not guarantee one
   assignment per player-fixture.

## Target structure

### Separate style from role

Use style rates only for the clustering:

```text
fg3a_share, ftr, fga_per_min, ast_per_min, ast_pct, ast_ratio,
tov_pct, oreb_pct, dreb_pct, stl_per_min, blk_per_min, pf_per_min
```

Exclude `min`, `start_rate`, efficiency, usage and team context. Put
`start_rate_5/10/20`, `ewm_min_3/8/20` and `depth_rank` in the minutes and
absorption model, where they belong.

The current defaults hide this choice: `features.rolling()` and
`features.design_matrix()` both default to `include_volume=True`, and
`scripts/pipeline.py` does not override it. Make the decision explicit and add
an ablation with and without volume. The README's "rate, not volume" rule
should match the production feature list.

### Separate fit from assign

- Fit the GMM once per season and freeze it.
- Assign responsibilities **daily** using the frozen model and strictly prior
  games.
- Publish `as_of` / `prediction_date` rows, not only the latest snapshot.
- If the assignment is used by `team_hierarchy`, it must be available before
  the 16:15 and 22:15 model runs, or computed inside that pipeline.

### Representation

- Keep the soft responsibilities as the representation.
- Use the label for reading only; never as a model input.
- Choose `k` by held-out log-likelihood, cluster-share sanity and stability,
  not BIC alone.
- Consider PCA and covariance-type ablations; calibrate confidence.

### Data contract

- `mod.player_archetype_base`: one row per
  `(prediction_date, game_date, game_id, player_id)` with responsibilities,
  label, confidence, entropy, feature values, model version and timestamp.
- `mod.player_archetype_current_vw`: latest assignment per player.
- `mod.player_archetype_profile_vw`: cluster profiles and descriptions.
- Coverage check: every rostered player with enough prior games gets an
  assignment; missing features are imputed or fall back to the last assignment.
- Add a per-run validation that responsibilities sum to one, keys are unique
  and current-season coverage is above a threshold.

## Integration with `team_hierarchy`

Add a similarity layer rather than a hard category:

```text
similarity(player, absent) = cosine or Bhattacharyya(responsibilities)
expected_vacated_minutes_similar_style =
    sum over absent teammates of
    (1 - p_play) * minutes_if_plays * similarity(player, absent)
```

- Use as-of responsibility vectors for every roster player on the prediction
  date.
- Add the feature to the absorption feature block in
  `team_hierarchy/src/team_hierarchy/absorption.py`.
- Ablate against the current position and start-rate features:
  - minutes MAE overall;
  - top-quartile vacated minutes;
  - role-change subset;
  - starter recall.
- Drop the feature if it adds nothing beyond `start_rate_5` and `depth_rank`.
  The project rule is to keep a feature only when an ablation shows a gain.

The motivating case is Sidy Cissoko on 11/17: predicted 19.8 minutes, played
32, with an archetype row from 11/2 labelled "Deep reserve." The new minutes
model reaches 23.1 pre-game. The archetype model is not the primary fix, but a
daily style-only responsibility vector is the right input for a "who replaces
this role?" feature.

## Evaluation

- Descriptive: `statyx.play_types` R², ARI stability, cluster shares,
  confidence/entropy, coverage.
- Predictive: minutes and absorption are the new test; usage is a known
  negative.
- Replacement test: for historical absences, does the most similar teammate
  gain more minutes? Compare across position groups and role-change cases.
- Calibration: confidence versus held-out cluster purity; test temperature
  scaling.

## Operations

- Daily assignment cadence aligned with `team_hierarchy`.
- Freshness gate: fail if there is no as-of assignment for the target date.
- Archive each seasonal fit; record checksum and model version in the output.
- Keep DDL in `nba/database/sql/views/mod/`; validate before publishing.
- Keep the taxonomy frozen during the season; refit in preseason and record the
  new stability and coverage results.

## Do not retest blindly

- Four usage formulations already rejected: own archetype, opponent archetype,
  teammate archetype mix and the interaction.
- Position versus archetype for `statyx.play_types` is already positive.
- Do not tune the model on one player's case; use the walk-forward minutes and
  absorption evaluation.

## Open questions

- Is the taxonomy descriptive or predictive? If predictive, style-only may lose
  a useful role signal; test it instead of assuming.
- Should role features be removed completely from the clustering? Recommended:
  yes, keep role in the minutes model.
- How should null rate features and players with short history be handled?
- Should archetypes replace position in absorption or augment it?
- Should the similarity be cosine on responsibilities or Bhattacharyya on the
  probability vectors? Test both; Bhattacharyya is the natural choice for
  distributions.

## File map

- `src/player_archetypes/features.py`: split style and role feature sets,
  imputation, explicit volume/efficiency flags.
- `src/player_archetypes/cluster.py`: fit, assign, stability, profile,
  calibration.
- `src/player_archetypes/similarity.py`: new similarity and as-of join helpers.
- `scripts/pipeline.py`: seasonal fit versus daily assignment.
- `scripts/job.py`: daily cadence, freshness and publish.
- `scripts/eval_minutes.py`: new minutes/absorption ablation.
- `nba/mod/team_hierarchy/src/team_hierarchy/absorption.py`: similarity
  features.
- `nba/database/sql/views/mod/player_archetype*`: public contracts.
