"""Soft role clustering.

The representation is deliberately soft. A hard cluster id is just `position`
with more categories -- it reimposes the discreteness that makes listed
position a poor description of the modern game. A Gaussian mixture gives each
player-game a vector of responsibilities ("62% primary creator, 31% off-ball
shooter"), which is what actually captures a player who does two jobs. The
argmax label exists for reading and plotting, not for modelling.

Every fit is time-boxed: `fit` is handed only rows strictly before a cutoff
date, so a cluster assignment for a game never depends on that game or later
ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class RoleModel:
    """A fitted role model plus the preprocessing it needs at apply time."""

    scaler: StandardScaler
    pca: PCA | None
    gmm: GaussianMixture
    names: list[str]
    cutoff: date | None

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        x = self.scaler.transform(df.select(self.names).to_numpy())
        return self.pca.transform(x) if self.pca is not None else x

    def responsibilities(self, df: pl.DataFrame) -> np.ndarray:
        """Soft assignment: one row per input, one column per role."""
        return self.gmm.predict_proba(self._matrix(df))

    def assign(self, df: pl.DataFrame, prefix: str = "role") -> pl.DataFrame:
        """Attach soft responsibilities, the argmax label and its confidence."""
        resp = self.responsibilities(df)
        k = resp.shape[1]
        return df.with_columns(
            [pl.Series(f"{prefix}_{i}", resp[:, i]) for i in range(k)]
            + [
                pl.Series(f"{prefix}_label", resp.argmax(axis=1)).cast(pl.Int32),
                pl.Series(f"{prefix}_confidence", resp.max(axis=1)),
                # Normalised entropy: 0 = pure role, 1 = evenly split across roles.
                pl.Series(
                    f"{prefix}_entropy",
                    -(resp * np.log(resp + 1e-12)).sum(axis=1) / np.log(k),
                ),
            ]
        )


def fit(
    df: pl.DataFrame,
    names: list[str],
    k: int,
    cutoff: date | None = None,
    n_pca: int | None = None,
    seed: int = 0,
    covariance_type: str = "full",
) -> RoleModel:
    """Fit a k-role mixture on rows strictly before `cutoff`."""
    train = df.filter(pl.col("game_date") < cutoff) if cutoff is not None else df
    if train.height < k * 50:
        raise ValueError(f"only {train.height} training rows for k={k}")

    scaler = StandardScaler().fit(train.select(names).to_numpy())
    x = scaler.transform(train.select(names).to_numpy())
    pca = None
    if n_pca is not None:
        pca = PCA(n_components=n_pca, random_state=seed).fit(x)
        x = pca.transform(x)
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=seed,
        n_init=4,
        max_iter=500,
    ).fit(x)
    return RoleModel(scaler=scaler, pca=pca, gmm=gmm, names=names, cutoff=cutoff)


def select_k(
    df: pl.DataFrame,
    names: list[str],
    ks: range | list[int],
    cutoff: date,
    seed: int = 0,
) -> pl.DataFrame:
    """Score each k by BIC on the fit window and log-likelihood held out after it.

    Held-out likelihood is the honest criterion: BIC rewards fit on data the
    model has seen, and with ~400k rows it will happily keep adding components.
    """
    held = df.filter(pl.col("game_date") >= cutoff)
    rows = []
    for k in ks:
        model = fit(df, names, k=k, cutoff=cutoff, seed=seed)
        x_held = model._matrix(held)
        rows.append(
            {
                "k": k,
                "bic": float(model.gmm.bic(model._matrix(df.filter(pl.col("game_date") < cutoff)))),
                "heldout_loglik": float(model.gmm.score(x_held)),
                "min_cluster_share": float(
                    np.bincount(
                        model.gmm.predict(x_held), minlength=k
                    ).min()
                    / x_held.shape[0]
                ),
            }
        )
    return pl.DataFrame(rows)


def stability(
    df: pl.DataFrame,
    names: list[str],
    k: int,
    cutoffs: list[date],
    seed: int = 0,
) -> pl.DataFrame:
    """Do the roles survive refitting on a different era?

    Each model is fitted on a different time window, then all of them label the
    same reference rows. Adjusted Rand Index compares those labellings. A role
    taxonomy that reshuffles whenever you move the cutoff is not describing the
    game, and is useless as a replacement for position.
    """
    ref = df.filter(pl.col("game_date") >= max(cutoffs))
    labels = {}
    for c in cutoffs:
        model = fit(df, names, k=k, cutoff=c, seed=seed)
        labels[c] = model.gmm.predict(model._matrix(ref))
    rows = []
    for i, a in enumerate(cutoffs):
        for b in cutoffs[i + 1 :]:
            rows.append(
                {
                    "cutoff_a": a,
                    "cutoff_b": b,
                    "ari": float(adjusted_rand_score(labels[a], labels[b])),
                }
            )
    return pl.DataFrame(rows)


def profile(assigned: pl.DataFrame, names: list[str], prefix: str = "role") -> pl.DataFrame:
    """Per-cluster feature means as z-scores, for naming the roles."""
    mu = assigned.select(names).mean().to_numpy()[0]
    sd = assigned.select(names).std().to_numpy()[0]
    out = (
        assigned.group_by(f"{prefix}_label")
        .agg([pl.col(c).mean().alias(c) for c in names] + [pl.len().alias("n")])
        .sort(f"{prefix}_label")
    )
    return out.with_columns(
        [
            ((pl.col(c) - mu[i]) / sd[i]).round(2).alias(c)
            for i, c in enumerate(names)
        ]
    )
