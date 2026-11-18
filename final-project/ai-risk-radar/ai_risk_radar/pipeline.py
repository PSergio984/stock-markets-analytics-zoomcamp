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
from ai_risk_radar.provenance import lookup_historical_provenance
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


def _compute_policy_hash(selected_threshold: float | None, settings: Settings) -> str:
    """Compute deterministic SHA256 of release policy and allocation rules."""
    policy_payload = {
        "selected_threshold": float(selected_threshold) if selected_threshold is not None else None,
        "weights_growth": [float(w) for w in settings.weights_growth],
        "weights_defensive": [float(w) for w in settings.weights_defensive],
        "weights_fixed_baseline": [float(w) for w in settings.weights_fixed_baseline],
        "cost_rate": float(settings.cost_rate),
        "tickers": list(settings.tickers),
    }
    return hashlib.sha256(json.dumps(policy_payload, sort_keys=True).encode("utf-8")).hexdigest()


def _compute_release_hash(model_hash: str, data_hash: str, policy_hash: str) -> str:
    """Compute compound SHA256 covering model weights, data manifest, and policy configuration."""
    compound = f"{model_hash}:{data_hash}:{policy_hash}"
    return hashlib.sha256(compound.encode("utf-8")).hexdigest()


def run_research(settings: Settings, data_mode: str = "cache") -> Path:
    """Fit models and select strategy on validation partition, leaving test untouched."""
    raw_dir = Path(settings.raw_dir)
    artifacts_dir = Path(settings.artifacts_dir)
    snapshot_dir = raw_dir / "snapshot"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Guard against silent holdout reset if existing release already consumed test partition
    existing_release_path = artifacts_dir / "release.json"
    previous_consumption = None
    if existing_release_path.is_file():
        try:
            existing_release_data = json.loads(existing_release_path.read_text(encoding="utf-8"))
            if "test_evaluation" in existing_release_data:
                previous_consumption = existing_release_data["test_evaluation"]
        except (json.JSONDecodeError, OSError):
            pass

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
    policy_hash = _compute_policy_hash(selected_threshold, settings)
    compound_release_hash = _compute_release_hash(model_hash, data_hash, policy_hash)

    release_payload = {
        "release_version": "1.0.0",
        "hashes": {
            "model": model_hash,
            "data": data_hash,
            "policy": policy_hash,
            "release": compound_release_hash,
        },
        "model_file": str(model_path.relative_to(artifacts_dir.parent)),
        "selected_strategy": selected_key,
        "selected_threshold": selected_threshold,
        "policy_hash": policy_hash,
        "release_hash": compound_release_hash,
        "probability_status": rf_bundle.metadata.get("probability_status"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "boundaries": splits.boundaries,
        "validation_metrics": selected_res.metrics,
    }
    if previous_consumption is not None:
        release_payload["previous_holdout_consumption"] = previous_consumption
        release_payload["holdout_status"] = "subsequent_research_holdout_previously_consumed"
    else:
        release_payload["holdout_status"] = "untouched"

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
        "model_hash": model_hash,
        "release_hash": compound_release_hash,
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
    is_reproduction = "test_evaluation" in release_data

    expected_model_hash = release_data.get("hashes", {}).get("model")
    expected_data_hash = release_data.get("hashes", {}).get("data")
    expected_policy_hash = release_data.get("hashes", {}).get("policy") or release_data.get("policy_hash")

    # If already evaluated, verify sufficient identity for reproduction equivalence
    if is_reproduction:
        expected_boundaries = release_data.get("boundaries")
        if not expected_boundaries or not isinstance(expected_boundaries, dict):
            legacy_provenance = lookup_historical_provenance(expected_model_hash or "", expected_data_hash or "")
            if legacy_provenance and legacy_provenance.expected_boundaries:
                expected_boundaries = legacy_provenance.expected_boundaries
            else:
                raise ValueError(
                    "Holdout test partition has already been consumed: release manifest has insufficient identity: missing 'boundaries'. "
                    "Cannot establish identical reproduction equivalence."
                )
        if "selected_threshold" not in release_data:
            raise ValueError(
                "Holdout test partition has already been consumed: release manifest has insufficient identity: missing 'selected_threshold'. "
                "Cannot establish identical reproduction equivalence."
            )
        if "test_metrics" not in release_data.get("test_evaluation", {}):
            raise ValueError(
                "Holdout test partition has already been consumed: release manifest has insufficient identity: missing 'test_metrics'. "
                "Cannot establish identical reproduction equivalence."
            )

    # Historical provenance resolution for legacy releases lacking embedded policy hashes
    legacy_provenance = None
    if not expected_policy_hash and is_reproduction:
        legacy_provenance = lookup_historical_provenance(expected_model_hash or "", expected_data_hash or "")
        if legacy_provenance:
            expected_policy_hash = legacy_provenance.policy_hash
        else:
            raise ValueError(
                "Holdout test partition has already been consumed: release manifest has insufficient identity: "
                "legacy release lacks embedded policy_hash and has no recognized source-backed provenance entry."
            )

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
            f"Hash mismatch on frozen model: expected {expected_model_hash}, got {actual_model_hash}. "
            "Altered model cannot silently reuse a consumed holdout."
        )

    settings, _project_dir = _resolve_pipeline_settings(rel_path, config_path=config_path, settings=settings)

    # Validate policy hash or source-backed historical provenance BEFORE snapshot loading / simulation
    selected_threshold = release_data.get("selected_threshold")
    actual_policy_hash = _compute_policy_hash(selected_threshold, settings)

    if expected_policy_hash and actual_policy_hash != expected_policy_hash:
        raise ValueError(
            f"Policy / configuration mismatch on release: expected policy hash {expected_policy_hash}, "
            f"got {actual_policy_hash}. Config/policy changes cannot silently reuse a consumed holdout as untouched."
        )

    # For legacy releases with source-backed provenance, verify explicit parameters before simulation
    if legacy_provenance is not None:
        p_params = legacy_provenance.policy_parameters
        if abs(settings.cost_rate - p_params["cost_rate"]) > 1e-7:
            raise ValueError(
                f"Policy parameter mismatch on reproduction: expected cost_rate={p_params['cost_rate']}, "
                f"got {settings.cost_rate}."
            )
        if tuple(settings.weights_growth) != tuple(p_params["weights_growth"]):
            raise ValueError(
                f"Policy parameter mismatch on reproduction: expected weights_growth={p_params['weights_growth']}, "
                f"got {settings.weights_growth}."
            )
        if tuple(settings.weights_defensive) != tuple(p_params["weights_defensive"]):
            raise ValueError(
                f"Policy parameter mismatch on reproduction: expected weights_defensive={p_params['weights_defensive']}, "
                f"got {settings.weights_defensive}."
            )
        if tuple(settings.weights_fixed_baseline) != tuple(p_params["weights_fixed_baseline"]):
            raise ValueError(
                f"Policy parameter mismatch on reproduction: expected weights_fixed_baseline={p_params['weights_fixed_baseline']}, "
                f"got {settings.weights_fixed_baseline}."
            )
        if tuple(settings.candidate_thresholds) != tuple(p_params["candidate_thresholds"]):
            raise ValueError(
                f"Policy parameter mismatch on reproduction: expected candidate_thresholds={p_params['candidate_thresholds']}, "
                f"got {settings.candidate_thresholds}."
            )
        if settings.random_seed != p_params["random_seed"]:
            raise ValueError(
                f"Policy parameter mismatch on reproduction: expected random_seed={p_params['random_seed']}, "
                f"got {settings.random_seed}."
            )

    # Load cached snapshot
    snapshot_dir = Path(settings.raw_dir) / "snapshot"
    bundle = load_snapshot(snapshot_dir)

    # Validate data manifest hash against release data hash if tracked
    actual_data_hash = "unknown"
    data_manifest_path = snapshot_dir / "manifest.json"
    if data_manifest_path.is_file():
        actual_data_hash = _compute_sha256(data_manifest_path)
        if expected_data_hash and expected_data_hash != "unknown" and actual_data_hash != expected_data_hash:
            raise ValueError(
                f"Hash mismatch on data snapshot: expected {expected_data_hash}, got {actual_data_hash}. "
                "Altered data cannot silently reuse a consumed holdout."
            )

    compound_release_hash = (
        release_data.get("hashes", {}).get("release")
        or release_data.get("release_hash")
        or _compute_release_hash(actual_model_hash, actual_data_hash, actual_policy_hash)
    )

    feat_set = build_features(bundle, settings)
    from ai_risk_radar.splits import make_splits
    splits = make_splits(feat_set, settings)

    # For reproduction runs, strictly reject altered partition dates or execution conventions before evaluation
    if is_reproduction:
        for b_key, exp_val in expected_boundaries.items():
            act_val = splits.boundaries.get(b_key)
            if exp_val and act_val and exp_val != act_val:
                raise ValueError(
                    f"Partition boundary mismatch on reproduction: expected {b_key}={exp_val}, got {act_val}. "
                    "Altered partition dates cannot silently reuse a consumed holdout."
                )
        if settings.target_horizon != 20:
            raise ValueError(
                f"Execution convention mismatch: expected target_horizon=20, got {settings.target_horizon}. "
                "Altered conventions cannot silently reuse a consumed holdout."
            )
        if abs(settings.target_threshold - (-0.05)) > 1e-6:
            raise ValueError(
                f"Execution convention mismatch: expected target_threshold=-0.05, got {settings.target_threshold}. "
                "Altered conventions cannot silently reuse a consumed holdout."
            )
        if settings.random_seed != 42:
            raise ValueError(
                f"Execution convention mismatch: expected random_seed=42, got {settings.random_seed}. "
                "Altered conventions cannot silently reuse a consumed holdout."
            )

    rf_bundle = joblib.load(model_path)

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

    # Trajectory equity math for both legacy post-funding and net initial capital conventions
    v_end = float(selected_res.daily["equity"].iloc[-1])
    initial_cap = float(settings.initial_capital)
    v_post_0 = float(selected_res.trades["v_post"].iloc[0]) if not selected_res.trades.empty else initial_cap

    net_initial_capital_ret = (v_end - initial_cap) / initial_cap
    legacy_post_funding_ret = (v_end - v_post_0) / v_post_0
    funding_impact_bps = (legacy_post_funding_ret - net_initial_capital_ret) * 10000.0

    if is_reproduction:
        # Verify identical performance against original consumed evaluation
        original_metrics = release_data["test_evaluation"].get("test_metrics", {})
        orig_ret = original_metrics.get("total_return")

        if orig_ret is not None:
            # Check if matching under legacy_post_funding convention
            if legacy_provenance and legacy_provenance.metric_convention == "legacy_post_funding":
                diff = abs(legacy_post_funding_ret - orig_ret)
                if diff > 1e-4:
                    raise ValueError(
                        f"Reproduction metric divergence under legacy post-funding convention: "
                        f"computed legacy_post_funding_total_return={legacy_post_funding_ret}, "
                        f"original={orig_ret} (diff={diff})"
                    )
                metric_convention = "legacy_post_funding"
            elif abs(selected_res.metrics["total_return"] - orig_ret) <= 1e-4:
                metric_convention = "net_initial_capital"
            elif abs(legacy_post_funding_ret - orig_ret) <= 1e-4:
                metric_convention = "legacy_post_funding"
            else:
                diff = abs(selected_res.metrics["total_return"] - orig_ret)
                raise ValueError(
                    f"Reproduction metric divergence: computed total_return={selected_res.metrics['total_return']}, "
                    f"legacy_post_funding_total_return={legacy_post_funding_ret}, original={orig_ret} (diff={diff})"
                )
        else:
            metric_convention = "net_initial_capital"

        metadata = {
            "pipeline_mode": "final_test_reproduction",
            "reproduction_status": "verified_match",
            "evaluation_partition": "test",
            "probability_status": rf_bundle.metadata.get("probability_status"),
            "fallback_reason": rf_bundle.metadata.get("reason"),
            "selected_threshold": selected_threshold,
            "model_hash": actual_model_hash,
            "release_hash": compound_release_hash,
            "cutoff_session": splits.boundaries["cutoff"],
            "diagnostics": diagnostics,
            "weights_growth": settings.weights_growth,
            "weights_defensive": settings.weights_defensive,
            "weights_fixed_baseline": settings.weights_fixed_baseline,
            "tickers": settings.tickers,
            "original_evaluated_at": release_data["test_evaluation"].get("evaluated_at"),
            "metric_convention": metric_convention,
            "net_initial_capital_total_return": net_initial_capital_ret,
            "legacy_post_funding_total_return": legacy_post_funding_ret,
            "funding_cost_impact_bps": funding_impact_bps,
        }

        report = build_report(all_results, test_risk, metadata)
        report_path = artifacts_dir / "reproduction_test_report.json"
        save_report(report, report_path)
        # Strictly DO NOT modify release.json or final_test_report.json
        return report_path

    metadata = {
        "pipeline_mode": "final_test",
        "evaluation_partition": "test",
        "probability_status": rf_bundle.metadata.get("probability_status"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "selected_threshold": selected_threshold,
        "model_hash": actual_model_hash,
        "release_hash": compound_release_hash,
        "cutoff_session": splits.boundaries["cutoff"],
        "diagnostics": diagnostics,
        "weights_growth": settings.weights_growth,
        "weights_defensive": settings.weights_defensive,
        "weights_fixed_baseline": settings.weights_fixed_baseline,
        "tickers": settings.tickers,
        "metric_convention": "net_initial_capital",
        "net_initial_capital_total_return": net_initial_capital_ret,
        "legacy_post_funding_total_return": legacy_post_funding_ret,
        "funding_cost_impact_bps": funding_impact_bps,
    }

    report = build_report(all_results, test_risk, metadata)
    report_path = artifacts_dir / "final_test_report.json"
    save_report(report, report_path)

    # Record holdout evaluation completion in release.json
    release_data["test_evaluation"] = {
        "evaluated_at": pd.Timestamp.now().isoformat(),
        "metric_convention": "net_initial_capital",
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
    actual_policy_hash = _compute_policy_hash(selected_threshold, settings)
    actual_data_hash = "unknown"
    data_manifest_path = snapshot_dir / "manifest.json"
    if data_manifest_path.is_file():
        actual_data_hash = _compute_sha256(data_manifest_path)

    compound_release_hash = (
        release_data.get("hashes", {}).get("release")
        or release_data.get("release_hash")
        or _compute_release_hash(actual_model_hash, actual_data_hash, actual_policy_hash)
    )

    metadata = {
        "pipeline_mode": "latest",
        "probability_status": rf_bundle.metadata.get("probability_status"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "selected_threshold": selected_threshold,
        "model_hash": actual_model_hash,
        "release_hash": compound_release_hash,
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
