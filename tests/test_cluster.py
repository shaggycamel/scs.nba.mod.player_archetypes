from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from player_archetypes import cluster, features


def make_design(players: int = 12, games: int = 25, seed: int = 0):
    """Synthetic design matrix with the same schema as the real pipeline."""
    rng = np.random.default_rng(seed)
    names = features.feature_names()
    start = date(2024, 1, 1)
    rows = []
    for player in range(players):
        for game in range(games):
            row = {
                "player_id": player,
                "player_name": f"Player {player}",
                "game_id": player * 10_000 + game,
                "game_date": start + timedelta(days=game),
                "season": "2024-25",
                "team_abbreviation": f"T{player % 3}",
            }
            row.update({name: float(rng.normal()) for name in names})
            rows.append(row)
    return pl.DataFrame(rows), names


def test_role_model_alias_preserves_the_old_pickle_path():
    assert cluster.RoleModel is cluster.ArchetypeModel


def test_assign_produces_probabilities():
    design, names = make_design()
    model = cluster.fit(design, names, k=3, seed=0)
    assigned = model.assign(design)

    probs = assigned.select([f"archetype_{i}" for i in range(3)]).to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert probs.min() >= 0.0
    assert assigned["archetype_label"].min() >= 0
    assert assigned["archetype_label"].max() < 3
    np.testing.assert_allclose(
        assigned["archetype_confidence"].to_numpy(), probs.max(axis=1), atol=1e-9
    )
    assert assigned["archetype_entropy"].min() >= -1e-9
    assert assigned["archetype_entropy"].max() <= 1.0 + 1e-9


def test_cutoff_uses_only_prior_rows():
    design, names = make_design()
    cutoff = date(2024, 1, 20)

    with_cutoff = cluster.fit(design, names, k=2, cutoff=cutoff, seed=0)
    prior_only = cluster.fit(
        design.filter(pl.col("game_date") < cutoff), names, k=2, cutoff=None, seed=0
    )

    np.testing.assert_allclose(
        with_cutoff.responsibilities(design),
        prior_only.responsibilities(design),
        atol=1e-12,
    )


def test_fit_rejects_too_few_rows():
    design, names = make_design(players=2, games=10)
    with pytest.raises(ValueError, match="training rows"):
        cluster.fit(design, names, k=3, seed=0)


def test_profile_returns_one_row_per_archetype():
    design, names = make_design()
    model = cluster.fit(design, names, k=3, seed=0)
    prof = cluster.profile(model.assign(design), names)

    assert prof.height == 3
    assert "n" in prof.columns
    assert set(prof["archetype_label"].to_list()) == {0, 1, 2}
