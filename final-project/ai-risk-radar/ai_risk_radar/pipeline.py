"""Pipeline orchestration: separate research, frozen final-test, and latest inference commands."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from ai_risk_radar.config import load_settings
from ai_risk_radar.contracts import Settings
from ai_risk_radar.data import fetch_prices, load_snapshot, save_snapshot
from ai_risk_radar.features import build_features
from ai_risk_radar.models import classification_report, fit_models, predict_risk
from ai_risk_radar.reporting import build_report, save_report
from ai_risk_radar.simulation import (
    benchmark_targets,
    select_strategy,
    simulate,
    weekly_targets,
)


def _compute_sha256(file_path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_research(settings: Settings, data_mode: str = "cache") -> Path:
    """Fit models and select strategy on validation partition, leaving test untouched."""
    raw_dir = Path(settings.raw_dir)
    artifacts_dir = Path(settings.artifacts_dir)
    snapshot_dir = raw_dir / "snapshot"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion / Data Loading
    if data_mode == "refresh":
        bundle = fetch_prices(settings)
        save_snapshot(bundle, snapshot_dir)
    elif data_mode == "cache":
        bundle = load_snapshot(snapshot_dir)
    else:
        raise ValueError(f"Invalid data_mode: '{data_mode}'. Must be 'refresh' or 'cache'.")

    # 2. Features and Splits
    feat_set = build_features(bundle, settings)
    from ai_risk_radar.splits import make_splits
    splits = make_splits(feat_set, settings)

    # 3. Model Training & Calibration
    models = fit_models(feat_set, splits, settings)
    rf_bundle = models["random_forest"]

    # Save model artifact
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "random_forest.joblib"
    joblib.dump(rf_bundle, model_path)
    model_hash = _compute_sha256(model_path)

    # 4. Strategy Simulation on Validation Partition Only
    val_frame = feat_set.frame.loc[splits.validation]
    val_prices = bundle.adjusted_close.loc[splits.validation]
    val_risk = predict_risk(rf_bundle, val_frame)

    candidates = {}
    for th in settings.candidate_thresholds:
        key = f"threshold_{th:.2f}"
        targets = weekly_targets(val_risk, splits.validation, settings, threshold=th)
        res = simulate(val_prices, targets, settings, cost_rate=settings.cost_rate)
        candidates[key] = res

    benchmarks = benchmark_targets(bundle, settings, sessions=splits.validation)
    base_res = simulate(
        val_prices,
        benchmarks["fixed_defensive_weekly"],
        settings,
        cost_rate=settings.cost_rate,
    )
    bnh_res = simulate(
        val_prices,
        benchmarks["growth_buy_and_hold"],
        settings,
        cost_rate=settings.cost_rate,
    )
    mom_res = simulate(
        val_prices,
        benchmarks["momentum_63"],
        settings,
        cost_rate=settings.cost_rate,
    )

    # 5. Selection
    selected_key = select_strategy(candidates, base_res)
    if selected_key is not None and rf_bundle.metadata.get("probability_status") == "calibrated":
        selected_res = candidates[selected_key]
        selected_threshold = float(selected_key.split("_")[1])
    else:
        selected_key = "fixed_defensive_weekly"
        selected_res = base_res
        selected_threshold = None

    # 6. Diagnostics on Validation
    y_val = feat_set.frame.loc[splits.validation, "downside20"]
    diagnostics = {
        "model_selection": {
            "cv_brier_score": rf_bundle.metadata.get("cv_brier_score"),
            "selected_max_depth": rf_bundle.metadata.get("selected_max_depth"),
            "selected_min_samples_leaf": rf_bundle.metadata.get("selected_min_samples_leaf"),
        },
        "validation_classification": classification_report(y_val, val_risk),
    }

    # 7. Release Manifest Persistence
    data_manifest_path = snapshot_dir / "manifest.json"
    data_hash = _compute_sha256(data_manifest_path) if data_manifest_path.is_file() else "unknown"

    release_payload = {
        "release_version": "1.0.0",
        "hashes": {
            "model": model_hash,
            "data": data_hash,
        },
        "model_file": str(model_path.relative_to(artifacts_dir.parent)),
        "selected_strategy": selected_key,
        "selected_threshold": selected_threshold,
        "probability_status": rf_bundle.metadata.get("probability_status"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "boundaries": splits.boundaries,
        "validation_metrics": selected_res.metrics,
    }
    release_path = artifacts_dir / "release.json"
    release_path.write_text(json.dumps(release_payload, indent=2), encoding="utf-8")

    # 8. Report Generation
    all_results = {
        "selected": selected_res,
        "growth_buy_and_hold": bnh_res,
        "fixed_defensive_weekly": base_res,
        "momentum_63": mom_res,
        **candidates,
    }
    metadata = {
        "pipeline_mode": "research",
        "evaluation_partition": "validation",
        "probability_status": rf_bundle.metadata.get("probability_status"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "selected_threshold": selected_threshold,
        "release_hash": model_hash,
        "cutoff_session": splits.boundaries["cutoff"],
        "diagnostics": diagnostics,
        "weights_growth": settings.weights_growth,
        "weights_defensive": settings.weights_defensive,
        "weights_fixed_baseline": settings.weights_fixed_baseline,
        "tickers": settings.tickers,
    }

    report = build_report(all_results, val_risk, metadata)
    report_path = artifacts_dir / "research_report.json"
    return save_report(report, report_path)


def _resolve_pipeline_settings(
    release_path: Path | str,
    config_path: Path | str | None = None,
    settings: Settings | None = None,
) -> tuple[Settings, Path]:
    """Resolve settings and project directory relative to release manifest or explicit config."""
    rel_path = Path(release_path).resolve()
    artifacts_dir = rel_path.parent
    project_dir = artifacts_dir.parent

    if settings is not None:
        return settings, project_dir

    if config_path is not None:
        cfg_file = Path(config_path).resolve()
    elif (project_dir / "config.json").is_file():
        cfg_file = project_dir / "config.json"
    elif Path("config.json").is_file():
        cfg_file = Path("config.json").resolve()
    else:
        raise FileNotFoundError(
            f"Configuration file not found for release at {rel_path}. Specify config_path explicitly."
        )

    loaded_settings = load_settings(cfg_file)

    raw_dir = Path(loaded_settings.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = (project_dir / raw_dir).resolve()

    processed_dir = Path(loaded_settings.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = (project_dir / processed_dir).resolve()

    artifacts = Path(loaded_settings.artifacts_dir)
    if not artifacts.is_absolute():
        artifacts = (project_dir / artifacts).resolve()

    resolved_settings = Settings(
        tickers=loaded_settings.tickers,
        acquisition_start=loaded_settings.acquisition_start,
        source=loaded_settings.source,
        target_horizon=loaded_settings.target_horizon,
        target_threshold=loaded_settings.target_threshold,
        weights_growth=loaded_settings.weights_growth,
        weights_defensive=loaded_settings.weights_defensive,
        weights_fixed_baseline=loaded_settings.weights_fixed_baseline,
        candidate_thresholds=loaded_settings.candidate_thresholds,
        cost_rate=loaded_settings.cost_rate,
        cost_sensitivities=loaded_settings.cost_sensitivities,
        random_seed=loaded_settings.random_seed,
        initial_capital=loaded_settings.initial_capital,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts,
    )
    return resolved_settings, project_dir


def run_final_test(
    release_manifest: Path | str,
    config_path: Path | str | None = None,
    settings: Settings | None = None,
) -> Path:
    """Evaluate frozen release model and selected strategy strictly once on untouched test split."""
    rel_path = Path(release_manifest).resolve()
    if not rel_path.is_file():
        raise FileNotFoundError(f"Release manifest not found: {rel_path}")

    release_data = json.loads(rel_path.read_text(encoding="utf-8"))
    expected_model_hash = release_data.get("hashes", {}).get("model")
    expected_data_hash = release_data.get("hashes", {}).get("data")
    if not expected_model_hash:
        raise ValueError("Invalid release manifest: missing model hash")

    # Model file resolution
    artifacts_dir = rel_path.parent
    model_path = artifacts_dir / "models" / "random_forest.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    actual_model_hash = _compute_sha256(model_path)
    if actual_model_hash != expected_model_hash:
        raise ValueError(
            f"Hash mismatch on frozen model: expected {expected_model_hash}, got {actual_model_hash}"
        )

    settings, _project_dir = _resolve_pipeline_settings(rel_path, config_path=config_path, settings=settings)

    # Load cached snapshot
    snapshot_dir = Path(settings.raw_dir) / "snapshot"
    bundle = load_snapshot(snapshot_dir)

    # Validate data manifest hash against release data hash if tracked
    if expected_data_hash and expected_data_hash != "unknown":
        data_manifest_path = snapshot_dir / "manifest.json"
        if data_manifest_path.is_file():
            actual_data_hash = _compute_sha256(data_manifest_path)
            if actual_data_hash != expected_data_hash:
                raise ValueError(
                    f"Hash mismatch on data snapshot: expected {expected_data_hash}, got {actual_data_hash}"
                )

    rf_bundle = joblib.load(model_path)

    feat_set = build_features(bundle, settings)
    from ai_risk_radar.splits import make_splits
    splits = make_splits(feat_set, settings)

    test_frame = feat_set.frame.loc[splits.test]
    test_prices = bundle.adjusted_close.loc[splits.test]
    test_risk = predict_risk(rf_bundle, test_frame)

    selected_threshold = release_data.get("selected_threshold")
    benchmarks = benchmark_targets(bundle, settings, sessions=splits.test)

    candidates = {}
    for th in settings.candidate_thresholds:
        key = f"threshold_{th:.2f}"
        targets = weekly_targets(test_risk, splits.test, settings, threshold=th)
        res = simulate(test_prices, targets, settings, cost_rate=settings.cost_rate)
        candidates[key] = res

    base_res = simulate(
        test_prices,
        benchmarks["fixed_defensive_weekly"],
        settings,
        cost_rate=settings.cost_rate,
    )
    bnh_res = simulate(
        test_prices,
        benchmarks["growth_buy_and_hold"],
        settings,
        cost_rate=settings.cost_rate,
    )
    mom_res = simulate(
        test_prices,
        benchmarks["momentum_63"],
        settings,
        cost_rate=settings.cost_rate,
    )

    if selected_threshold is not None:
        selected_res = candidates.get(f"threshold_{selected_threshold:.2f}", base_res)
    else:
        selected_res = base_res

    # Test diagnostics (only on sessions where label is known, excluding unlabelled last 20)
    y_test = feat_set.frame.loc[splits.test, "downside20"]
    diagnostics = {
        "test_classification": classification_report(y_test, test_risk),
    }

    all_results = {
        "selected": selected_res,
        "growth_buy_and_hold": bnh_res,
        "fixed_defensive_weekly": base_res,
        "momentum_63": mom_res,
        **candidates,
    }

    metadata = {
        "pipeline_mode": "final_test",
        "evaluation_partition": "test",
        "probability_status": rf_bundle.metadata.get("probability_status"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "selected_threshold": selected_threshold,
        "release_hash": actual_model_hash,
        "cutoff_session": splits.boundaries["cutoff"],
        "diagnostics": diagnostics,
        "weights_growth": settings.weights_growth,
        "weights_defensive": settings.weights_defensive,
        "weights_fixed_baseline": settings.weights_fixed_baseline,
        "tickers": settings.tickers,
    }

    report = build_report(all_results, test_risk, metadata)
    report_path = artifacts_dir / "final_test_report.json"
    save_report(report, report_path)

    # Record holdout evaluation completion in release.json
    release_data["test_evaluation"] = {
        "evaluated_at": pd.Timestamp.now().isoformat(),
        "test_metrics": selected_res.metrics,
    }
    rel_path.write_text(json.dumps(release_data, indent=2), encoding="utf-8")

    return report_path


def run_latest(
    release_manifest: Path | str,
    data_mode: str = "cache",
    config_path: Path | str | None = None,
    settings: Settings | None = None,
) -> Path:
    """Generate latest real-time risk estimate and allocation suggestion using frozen release model."""
    rel_path = Path(release_manifest).resolve()
    if not rel_path.is_file():
        raise FileNotFoundError(f"Release manifest not found: {rel_path}")

    release_data = json.loads(rel_path.read_text(encoding="utf-8"))
    expected_model_hash = release_data.get("hashes", {}).get("model")
    if not expected_model_hash:
        raise ValueError("Invalid release manifest: missing model hash")

    artifacts_dir = rel_path.parent
    model_path = artifacts_dir / "models" / "random_forest.joblib"
    actual_model_hash = _compute_sha256(model_path)
    if actual_model_hash != expected_model_hash:
        raise ValueError("Hash mismatch on frozen demonstration model")

    rf_bundle = joblib.load(model_path)
    settings, _project_dir = _resolve_pipeline_settings(rel_path, config_path=config_path, settings=settings)
    snapshot_dir = Path(settings.raw_dir) / "snapshot"

    if data_mode == "refresh":
        bundle = fetch_prices(settings)
        save_snapshot(bundle, snapshot_dir)
    else:
        bundle = load_snapshot(snapshot_dir)

    feat_set = build_features(bundle, settings)
    latest_frame = feat_set.frame.iloc[-1:]
    latest_risk = predict_risk(rf_bundle, latest_frame)

    as_of_session = feat_set.frame.index[-1]
    now_ny = pd.Timestamp.now(tz="UTC").tz_convert("America/New_York")
    staleness = int((now_ny.normalize().tz_localize(None) - as_of_session).days)

    selected_threshold = release_data.get("selected_threshold")

    metadata = {
        "pipeline_mode": "latest",
        "probability_status": rf_bundle.metadata.get("probability_status"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "selected_threshold": selected_threshold,
        "release_hash": actual_model_hash,
        "cutoff_session": as_of_session.strftime("%Y-%m-%d"),
        "staleness_sessions": staleness,
        "refresh_status": "ok",
        "weights_growth": settings.weights_growth,
        "weights_defensive": settings.weights_defensive,
        "weights_fixed_baseline": settings.weights_fixed_baseline,
        "tickers": settings.tickers,
    }

    report = build_report({}, latest_risk, metadata)
    report_path = artifacts_dir / "latest_report.json"
    return save_report(report, report_path)
