"""Walk-forward archetype assignments for historical evaluation.

Each season is assigned by a model fitted only on games before that season, so
a 2024-25 row never sees 2024-25 data. This is the honest input for the
team-hierarchy absorption ablation; the production table only carries the
current season.

Usage:
    uv run --group data python -m scripts.backfill_assignments \
        --from-season 2021-22 --to-season 2025-26
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from player_archetypes import cluster, features
from player_archetypes.data import live, load


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-con", default="jetson-read")
    parser.add_argument("--out", type=Path, default=Path("artifacts/backfill"))
    parser.add_argument("--from-season", default="2021-22")
    parser.add_argument("--to-season", default="2025-26")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument("--sample", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    from scs_hub import SportsHub

    hub = SportsHub(db_con=args.db_con)
    log(f"reading live tables with [{args.db_con}]")
    box = load.box_scores_from_frames(live.read_box_scores(hub.db), live.read_schedule(hub.db))
    box = box.filter(pl.col("season_type") == "Regular Season")

    pg = features.usage_history(features.per_game(box))
    rolled = features.rolling(pg, window=args.window, min_periods=args.min_periods)
    design, names = features.design_matrix(rolled, min_prior_games=args.min_prior_games)
    log(f"design matrix: {design.height:,} rows x {len(names)} features")

    seasons = sorted(
        season
        for season in design["season"].unique().to_list()
        if season is not None and args.from_season <= season <= args.to_season
    )
    if not seasons:
        print("error: no seasons in the requested range", flush=True)
        return 2

    frames: list[pl.DataFrame] = []
    metadata: dict[str, dict] = {}
    for season in seasons:
        season_rows = design.filter(pl.col("season") == season)
        if season_rows.is_empty():
            continue
        cutoff = season_rows["game_date"].min()
        train = design.filter(pl.col("game_date") < cutoff)
        if args.sample and train.height > args.sample:
            train = train.sample(args.sample, seed=args.seed)
        log(f"{season}: fit on {train.height:,} rows before {cutoff}")
        model = cluster.fit(
            train,
            names,
            k=args.k,
            cutoff=cutoff,
            seed=args.seed,
            window=args.window,
            min_periods=args.min_periods,
            min_prior_games=args.min_prior_games,
            season_type="Regular Season",
        )
        archive = out_dir / "models" / str(cutoff)
        archive.mkdir(parents=True, exist_ok=True)
        model_path = archive / "model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        checksum = _sha256(model_path)

        assigned = model.assign(season_rows)
        archetype_cols = [f"archetype_{i}" for i in range(model.gmm.n_components)]
        frames.append(
            assigned.select(
                "player_id",
                "game_date",
                "season",
                "features_through",
                "archetype_label",
                "archetype_confidence",
                "archetype_entropy",
                *archetype_cols,
            ).with_columns(
                pl.lit(str(cutoff)).alias("model_cutoff"),
                pl.lit(checksum).alias("model_checksum"),
            )
        )
        metadata[season] = {
            "cutoff": str(cutoff),
            "checksum": checksum,
            "fit_rows": train.height,
            "assigned_rows": assigned.height,
            "k": int(model.gmm.n_components),
        }

    if not frames:
        print("error: no assignments produced", flush=True)
        return 2
    combined = pl.concat(frames, how="vertical").sort(["game_date", "player_id"])
    target = out_dir / "assignments.parquet"
    tmp = target.with_name(target.name + ".tmp")
    combined.write_parquet(tmp)
    tmp.replace(target)
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "features": names,
                "k": args.k,
                "seasons": metadata,
                "rows": combined.height,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"wrote {combined.height:,} rows to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
