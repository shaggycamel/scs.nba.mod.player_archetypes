"""Fit or load the archetype model, assign archetypes, and publish the artifacts.

This is the scheduled entry point. It reads the live database through scs-hub's
read-only connection; there is no offline snapshot path.

Usage:
    uv run --group data python -m scripts.pipeline    # live DB, saved model
    uv run --group data python -m scripts.pipeline --refit
    uv run --group data python -m scripts.pipeline --cutoff 2026-10-20 --refit

Outputs, under ``artifacts/pipeline/`` by default:

    model.pkl              fitted scaler/PCA/mixture (written only on a refit)
    model.json             version, checksum and fit settings for the artifact
    assignments.parquet    one row per player-game with archetype responsibilities
    run.json               provenance, summary and cluster profiles for this run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl

from player_archetypes import cluster, features, naming
from player_archetypes.data import live, load

ROOT = Path(__file__).resolve().parents[1]
ARCHETYPE_COLUMNS = ("archetype_label", "archetype_confidence", "archetype_entropy")


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
        default="jetson-read",
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
    parser.add_argument("--k", type=int, default=10, help="number of archetypes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=10, help="trailing games per feature")
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument(
        "--include-role",
        action="store_true",
        help="add start_rate and min to the clustering inputs (ablation only)",
    )
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


def _save_model(model: cluster.ArchetypeModel, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_model(path: Path) -> cluster.ArchetypeModel:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_meta(
    model: cluster.ArchetypeModel,
    *,
    version: str,
    checksum: str,
    names: list[str],
) -> dict:
    return {
        "model_version": version,
        "model_checksum": checksum,
        "cutoff": str(model.cutoff) if model.cutoff is not None else None,
        "k": int(model.gmm.n_components),
        "features": names,
        "window": model.window,
        "min_periods": model.min_periods,
        "min_prior_games": model.min_prior_games,
        "season_type": model.season_type,
        "fit_rows": model.fit_rows,
        "seed": model.seed,
    }


def _archive_model(model_path: Path, meta: dict, out_dir: Path) -> None:
    """Keep every seasonal fit, not just the mutable current-model file."""
    archive_dir = out_dir / "models" / meta["model_version"]
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, archive_dir / model_path.name)
    _atomic_json(meta, archive_dir / "model.json")
    _atomic_json(meta, model_path.with_name("model.json"))


def _load_model_meta(model_path: Path) -> dict:
    meta_path = model_path.with_name("model.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}; refit once to create the model sidecar")
    meta = json.loads(meta_path.read_text())
    actual = _file_sha256(model_path)
    if actual != meta["model_checksum"]:
        raise ValueError(
            f"{model_path} checksum {actual[:12]} does not match "
            f"model.json {meta['model_checksum'][:12]}"
        )
    return meta


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
        log(f"regular season only: {box.height:,} of {before:,} played player-games retained")
    season_type = "Regular Season" if args.regular_season_only else None

    pg = features.usage_history(features.per_game(box))
    rolled = features.rolling(
        pg,
        window=args.window,
        min_periods=args.min_periods,
        include_role=args.include_role,
    )
    design, names = features.design_matrix(
        rolled,
        include_role=args.include_role,
        min_prior_games=args.min_prior_games,
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
        checksum = _file_sha256(model_path)
        version = f"{cutoff}.{checksum[:8]}"
        model_meta = _model_meta(model, version=version, checksum=checksum, names=names)
        _archive_model(model_path, model_meta, out_dir)
        mode = "refit"
        log(f"wrote {version} to {model_path} and archived it")
    else:
        log(f"loading model from {model_path}")
        model = _load_model(model_path)
        model_meta = _load_model_meta(model_path)
        version = model_meta["model_version"]
        checksum = model_meta["model_checksum"]
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
            f"  {key}: model {getattr(model, key, None)!r}, runtime {value!r}"
            for key, value in expected.items()
            if getattr(model, key, None) != value
        ]
        if mismatches:
            print(
                "error: saved model was built with different settings\n" + "\n".join(mismatches),
                flush=True,
            )
            return 2
        fit_rows = getattr(model, "fit_rows", None)
        mode = "assign"

    log("assigning archetypes")
    assigned = model.assign(design)
    archetype_cols = [f"archetype_{i}" for i in range(model.gmm.n_components)]
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
            "features_through",
        )
        if column in assigned.columns
    ]
    published = assigned.select([*keys, *ARCHETYPE_COLUMNS, *archetype_cols, *names]).with_columns(
        pl.lit(model_meta["cutoff"]).cast(pl.String).str.to_date().alias("model_cutoff"),
        pl.lit(version).alias("model_version"),
        pl.lit(checksum).alias("model_checksum"),
    )
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
    profiles = naming.describe_archetypes(cluster.profile(assigned, names))
    _atomic_parquet(published, assignments_path)

    shares = (
        published.group_by("archetype_label")
        .agg(pl.len().alias("n"))
        .with_columns(share=pl.col("n") / published.height)
        .sort("archetype_label")
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
            "version": version,
            "checksum": checksum,
            "k": int(model.gmm.n_components),
            "cutoff": model_meta["cutoff"],
            "features": names,
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
            "players": int(published["player_id"].n_unique()),
            "season": (str(published["season"].max()) if args.current_season_only else None),
            "features_through_max": str(published["features_through"].max()),
            "mean_confidence": float(published["archetype_confidence"].mean()),
            "frac_ambiguous": float((published["archetype_confidence"] < 0.6).mean()),
            "cluster_shares": {
                str(row["archetype_label"]): round(row["share"], 6)
                for row in shares.iter_rows(named=True)
            },
        },
        "profiles": profiles.to_dicts(),
        "outputs": {
            "assignments": str(assignments_path),
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
