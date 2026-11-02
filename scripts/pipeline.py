"""Fit or load the archetype model, assign roles, and publish the artifacts.

This is the scheduled entry point. It starts from an existing Parquet snapshot
(refreshed separately), so it never touches the database.

Usage:
    uv run python -m scripts.pipeline                 # load the saved model, assign
    uv run python -m scripts.pipeline --refit         # fit a new model first
    uv run python -m scripts.pipeline --cutoff 2026-11-01 --refit

Outputs, under ``artifacts/pipeline/`` by default:

    model.pkl              fitted scaler/PCA/mixture (written only on a refit)
    assignments.parquet    one row per player-game with role responsibilities
    role_profiles.parquet  per-role feature means as z-scores
    run.json               provenance and summary for this run
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl

from player_archetypes import cluster, features
from player_archetypes.data import load
from player_archetypes.data.ids import DATA

ROOT = Path(__file__).resolve().parents[1]
ROLE_COLUMNS = ("role_label", "role_confidence", "role_entropy")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA,
        help="Parquet snapshot root containing nba/, statyx/ and util/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/pipeline"),
        help="directory for the model and published artifacts",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="model path; defaults to <out>/model.pkl",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="fit a new model even if one already exists",
    )
    parser.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        default=None,
        help="fit on games before this date (implies --refit)",
    )
    parser.add_argument("--k", type=int, default=10, help="number of roles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=20, help="trailing games per feature")
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument(
        "--sample",
        type=int,
        default=120_000,
        help="maximum rows used to fit the mixture (0 fits all)",
    )
    parser.add_argument(
        "--require-snapshot",
        action="store_true",
        help="fail unless the data directory has a _snapshot.json manifest",
    )
    parser.add_argument(
        "--max-snapshot-age-days",
        type=int,
        default=None,
        help="fail if the snapshot manifest is older than this many days",
    )
    return parser.parse_args(argv)


def _atomic_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _save_model(model: cluster.RoleModel, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_model(path: Path) -> cluster.RoleModel:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_version() -> str | None:
    try:
        return version("player-archetypes")
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data).expanduser().resolve()
    box_path = data_dir / "nba" / "player_box_score.parquet"
    if not box_path.exists():
        print(f"error: no box scores at {box_path}", flush=True)
        return 2

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).expanduser() if args.model else out_dir / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = data_dir / "_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        log(
            f"snapshot: last played {snapshot.get('max_game_date')}, "
            f"fetched {snapshot.get('fetched_at')}"
        )
        if args.max_snapshot_age_days is not None:
            fetched = datetime.fromisoformat(snapshot["fetched_at"])
            age_days = (datetime.now(UTC) - fetched).days
            if age_days > args.max_snapshot_age_days:
                print(
                    f"error: snapshot is {age_days} days old "
                    f"(limit {args.max_snapshot_age_days}); refresh it first",
                    flush=True,
                )
                return 2
    elif args.require_snapshot:
        print(f"error: no snapshot manifest at {snapshot_path}", flush=True)
        return 2
    else:
        log("no _snapshot.json; assuming the data directory was refreshed externally")

    log(f"loading box scores from {data_dir}")
    box = load.box_scores(data_dir)
    log(f"{box.height:,} played player-games")

    pg = features.usage_history(features.per_game(box))
    rolled = features.rolling(pg, window=args.window, min_periods=args.min_periods)
    design, names = features.design_matrix(
        rolled, min_prior_games=args.min_prior_games
    )
    if design.height == 0:
        print("error: the design matrix is empty after the prior-games filter", flush=True)
        return 2
    log(
        f"design matrix: {design.height:,} rows x {len(names)} features, "
        f"{design['game_date'].min()} to {design['game_date'].max()}"
    )

    refit = args.refit or args.cutoff is not None or not model_path.exists()
    fit_rows: int | None = None
    if refit:
        cutoff = args.cutoff or date.today()
        train = design.filter(pl.col("game_date") < cutoff)
        if train.height == 0:
            print(f"error: no games before cutoff {cutoff}", flush=True)
            return 2
        if args.sample and train.height > args.sample:
            train = train.sample(args.sample, seed=args.seed)
        log(f"fitting k={args.k} on {train.height:,} rows before {cutoff}")
        model = cluster.fit(train, names, k=args.k, cutoff=cutoff, seed=args.seed)
        _save_model(model, model_path)
        fit_rows = train.height
        mode = "refit"
        log(f"wrote model to {model_path}")
    else:
        log(f"loading model from {model_path}")
        model = _load_model(model_path)
        if model.names != names:
            print(
                "error: saved model features do not match the current design matrix\n"
                f"  model:   {model.names}\n"
                f"  current: {names}",
                flush=True,
            )
            return 2
        mode = "assign"

    log("assigning roles")
    assigned = model.assign(design)
    role_cols = [f"role_{i}" for i in range(model.gmm.n_components)]
    keys = [
        column
        for column in (
            "player_id",
            "player_name",
            "game_id",
            "game_date",
            "season",
            "team_abbreviation",
            "prior_games",
        )
        if column in assigned.columns
    ]
    published = assigned.select([*keys, *ROLE_COLUMNS, *role_cols])

    assignments_path = out_dir / "assignments.parquet"
    profiles_path = out_dir / "role_profiles.parquet"
    _atomic_parquet(published, assignments_path)
    _atomic_parquet(cluster.profile(assigned, names), profiles_path)

    shares = (
        assigned.group_by("role_label")
        .agg(pl.len().alias("n"))
        .with_columns(share=pl.col("n") / assigned.height)
        .sort("role_label")
    )
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "git_sha": _git_sha(),
        "package_version": _package_version(),
        "data_dir": str(data_dir),
        "design": {
            "rows": design.height,
            "players": int(design["player_id"].n_unique()),
            "date_min": str(design["game_date"].min()),
            "date_max": str(design["game_date"].max()),
            "features": names,
            "window": args.window,
            "min_periods": args.min_periods,
            "min_prior_games": args.min_prior_games,
        },
        "model": {
            "path": str(model_path),
            "k": int(model.gmm.n_components),
            "cutoff": str(model.cutoff) if model.cutoff is not None else None,
            "seed": args.seed,
            "covariance_type": model.gmm.covariance_type,
            "fit_rows": fit_rows,
        },
        "assignments": {
            "rows": published.height,
            "mean_confidence": float(assigned["role_confidence"].mean()),
            "frac_ambiguous": float((assigned["role_confidence"] < 0.6).mean()),
            "cluster_shares": {
                str(row["role_label"]): round(row["share"], 6)
                for row in shares.iter_rows(named=True)
            },
        },
        "outputs": {
            "assignments": str(assignments_path),
            "role_profiles": str(profiles_path),
            "model": str(model_path),
        },
    }
    _atomic_json(meta, out_dir / "run.json")

    log(
        f"published {published.height:,} assignments, "
        f"k={model.gmm.n_components}, mean confidence "
        f"{meta['assignments']['mean_confidence']:.3f} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
