"""Fit or load the archetype model, assign roles, and publish the artifacts.

This is the scheduled entry point. It reads the live database through scs-hub's
read-only connection; there is no offline snapshot path.

Usage:
    uv run --group data python -m scripts.pipeline    # live DB, saved model
    uv run --group data python -m scripts.pipeline --refit
    uv run --group data python -m scripts.pipeline --cutoff 2026-10-20 --refit

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
from player_archetypes.data import live, load

ROOT = Path(__file__).resolve().parents[1]
ROLE_COLUMNS = ("role_label", "role_confidence", "role_entropy")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def regular_season_only(box: pl.DataFrame, enabled: bool) -> pl.DataFrame:
    """Keep only regular-season rows before any rolling feature is built."""
    if not enabled:
        return box
    return box.filter(pl.col("season_type") == "Regular Season")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-con",
        default="cockroach-read",
        help="credentials.ini section used for live reads",
    )
    parser.add_argument(
        "--max-data-age-days",
        type=int,
        default=7,
        help="fail if the newest played game is older than this many days; 0 disables",
    )
    parser.add_argument(
        "--regular-season-only",
        action="store_true",
        help="exclude preseason, All-Star and playoff games before building features",
    )
    parser.add_argument(
        "--current-season-only",
        action="store_true",
        help="write only the latest season to assignments.parquet and run.json",
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
    parser.add_argument("--window", type=int, default=10, help="trailing games per feature")
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument(
        "--sample",
        type=int,
        default=120_000,
        help="maximum rows used to fit the mixture (0 fits all)",
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

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).expanduser() if args.model else out_dir / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    refit = args.refit or args.cutoff is not None
    if not refit and not model_path.exists():
        print(
            f"error: no saved model at {model_path}; pass --refit to train one",
            flush=True,
        )
        return 2

    try:
        from scs_hub import SportsHub
    except ImportError:
        print(
            "error: the pipeline requires scs-hub; run with "
            "`uv run --group data python -m scripts.pipeline`",
            flush=True,
        )
        return 2

    log(f"reading live tables with [{args.db_con}]")
    hub = SportsHub(db_con=args.db_con)
    raw_box = live.read_box_scores(hub.db)
    schedule = live.read_schedule(hub.db)
    box = load.box_scores_from_frames(raw_box, schedule)
    source = {
        "source": "live",
        "db_con": args.db_con,
        "raw_box_rows": raw_box.height,
    }

    log(f"{box.height:,} played player-games")
    latest_game = box["game_date"].max()
    data_age_days = (
        (datetime.now(UTC).date() - latest_game).days if latest_game is not None else None
    )
    if latest_game is None:
        print("error: the live database returned no played games", flush=True)
        return 2
    log(f"latest played game: {latest_game} ({data_age_days} days old)")
    if args.max_data_age_days and data_age_days > args.max_data_age_days:
        print(
            f"error: live data is {data_age_days} days old "
            f"(limit {args.max_data_age_days}); the upstream update may have failed",
            flush=True,
        )
        return 2
    source["max_game_date"] = str(latest_game)
    source["data_age_days"] = data_age_days

    if args.regular_season_only:
        before = box.height
        box = regular_season_only(box, True)
        log(
            f"regular season only: {box.height:,} of {before:,} "
            f"played player-games retained"
        )
    season_type = "Regular Season" if args.regular_season_only else None

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
        model = cluster.fit(
            train,
            names,
            k=args.k,
            cutoff=cutoff,
            seed=args.seed,
            window=args.window,
            min_periods=args.min_periods,
            min_prior_games=args.min_prior_games,
            season_type=season_type,
        )
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
        expected = {
            "window": args.window,
            "min_periods": args.min_periods,
            "min_prior_games": args.min_prior_games,
            "season_type": season_type,
        }
        mismatches = [
            f"  {key}: model {getattr(model, key, None)!r}, "
            f"runtime {value!r}"
            for key, value in expected.items()
            if getattr(model, key, None) != value
        ]
        if mismatches:
            print(
                "error: saved model was built with different settings\n"
                + "\n".join(mismatches),
                flush=True,
            )
            return 2
        fit_rows = getattr(model, "fit_rows", None)
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
    published = assigned.select([*keys, *ROLE_COLUMNS, *role_cols, *names])
    if args.current_season_only:
        seasons = published["season"].drop_nulls()
        if seasons.len() == 0:
            print("error: no season available for --current-season-only", flush=True)
            return 2
        current_season = seasons.max()
        before = published.height
        published = published.filter(pl.col("season") == current_season)
        log(
            f"current season only ({current_season}): "
            f"{published.height:,} of {before:,} rows retained"
        )

    assignments_path = out_dir / "assignments.parquet"
    profiles_path = out_dir / "role_profiles.parquet"
    _atomic_parquet(published, assignments_path)
    _atomic_parquet(cluster.profile(assigned, names), profiles_path)

    shares = (
        published.group_by("role_label")
        .agg(pl.len().alias("n"))
        .with_columns(share=pl.col("n") / published.height)
        .sort("role_label")
    )
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "git_sha": _git_sha(),
        "package_version": _package_version(),
        "source": source,
        "design": {
            "rows": design.height,
            "players": int(design["player_id"].n_unique()),
            "date_min": str(design["game_date"].min()),
            "date_max": str(design["game_date"].max()),
            "features": names,
            "season_type": season_type,
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
            "window": getattr(model, "window", None),
            "min_periods": getattr(model, "min_periods", None),
            "min_prior_games": getattr(model, "min_prior_games", None),
            "season_type": getattr(model, "season_type", None),
        },
        "assignments": {
            "rows": published.height,
            "season": (
                str(published["season"].max())
                if args.current_season_only
                else None
            ),
            "mean_confidence": float(published["role_confidence"].mean()),
            "frac_ambiguous": float((published["role_confidence"] < 0.6).mean()),
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
