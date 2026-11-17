"""Deterministic unit tests for pipeline commands: research, final-test, and latest."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.contracts import Settings
from ai_risk_radar.pipeline import (
    run_final_test,
    run_latest,
    run_research,
)


@pytest.fixture
def pipeline_env(tmp_path: Path):
    from ai_risk_radar.config import load_settings
    from ai_risk_radar.contracts import DataBundle
    from ai_risk_radar.data import save_snapshot

    cfg = {
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
        "acquisition_start": "2016-01-01",
        "source": "yfinance",
        "target_horizon": 20,
        "target_threshold": -0.05,
        "weights": {
            "growth": [0.35, 0.35, 0.20, 0.10],
            "defensive": [0.10, 0.15, 0.35, 0.40],
            "fixed_baseline": [0.20, 0.20, 0.20, 0.40],
        },
        "candidate_thresholds": [0.20, 0.35, 0.50],
        "cost_rate": 0.001,
        "cost_sensitivities": [0.0005, 0.0010, 0.0025],
        "random_seed": 42,
        "initial_capital": 10000.0,
        "data_paths": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg))
    settings = load_settings(cfg_file)

    # 10 years of deterministic data
    sessions = pd.date_range("2016-01-04", "2026-11-16", freq="B")
    n = len(sessions)
    np.random.seed(42)

    # Synthetic realistic price drift
    soxx = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
    xlk = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.012))
    spy = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.008))
    shy = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.001))

    prices = pd.DataFrame(
        {"SOXX": soxx, "XLK": xlk, "SPY": spy, "SHY": shy},
        index=sessions,
    )
    volume = pd.DataFrame(10000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])

    bundle = DataBundle(
        adjusted_close=prices,
        volume=volume,
        sessions=sessions,
        manifest={"source": "test_fixture", "cutoff_session": "2026-11-16"},
    )
    save_snapshot(bundle, Path(settings.raw_dir) / "snapshot")

    return settings, tmp_path


def test_research_does_not_evaluate_test_outcomes(pipeline_env):
    """Research mode fits and selects on validation, leaving test untouched."""
    settings, _tmp_path = pipeline_env

    report_path = run_research(settings, data_mode="cache")
    assert report_path.is_file()

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_data["performance"]["evaluation_partition"] == "validation"

    release_path = Path(settings.artifacts_dir) / "release.json"
    assert release_path.is_file()
    release_data = json.loads(release_path.read_text(encoding="utf-8"))
    assert "test_evaluation" not in release_data


def test_release_hash_change_blocks_test_reuse(pipeline_env):
    """Tampering with release.json model hash blocks final-test evaluation."""
    settings, _tmp_path = pipeline_env

    run_research(settings, data_mode="cache")
    release_path = Path(settings.artifacts_dir) / "release.json"
    release_data = json.loads(release_path.read_text(encoding="utf-8"))

    # Tamper with model hash
    release_data["hashes"]["model"] = "bad_tampered_hash_value"
    release_path.write_text(json.dumps(release_data))

    with pytest.raises(ValueError, match="Hash mismatch"):
        run_final_test(release_path)


def test_latest_does_not_fit_models(pipeline_env):
    """Latest mode uses the frozen release model and does not retrain."""
    settings, _tmp_path = pipeline_env
    run_research(settings, data_mode="cache")
    release_path = Path(settings.artifacts_dir) / "release.json"

    # Record model file mtime
    model_path = Path(settings.artifacts_dir) / "models" / "random_forest.joblib"
    mtime_before = model_path.stat().st_mtime

    latest_report_path = run_latest(release_path, data_mode="cache")
    assert latest_report_path.is_file()

    mtime_after = model_path.stat().st_mtime
    assert mtime_before == mtime_after, "Latest mode retrained or modified the model file!"


def test_incomplete_refresh_preserves_previous_report(pipeline_env):
    """An interrupted or failed refresh preserves the last valid report."""
    settings, tmp_path = pipeline_env
    run_research(settings, data_mode="cache")
    release_path = Path(settings.artifacts_dir) / "release.json"

    report_1 = run_latest(release_path, data_mode="cache")
    content_before = report_1.read_text(encoding="utf-8")

    # Invalidate data path temporarily to simulate failure
    with pytest.raises((RuntimeError, ValueError, OSError, FileNotFoundError)):
        # Simulate network or ingestion failure in refresh mode with invalid config
        bad_settings = Settings(
            tickers=("SOXX", "XLK", "SPY", "SHY"),
            acquisition_start="2099-01-01",  # Future start will fail
            source="invalid_source",
            target_horizon=20,
            target_threshold=-0.05,
            weights_growth=(0.35, 0.35, 0.20, 0.10),
            weights_defensive=(0.10, 0.15, 0.35, 0.40),
            weights_fixed_baseline=(0.20, 0.20, 0.20, 0.40),
            candidate_thresholds=(0.20, 0.35, 0.50),
            cost_rate=0.001,
            cost_sensitivities=(0.0005, 0.0010, 0.0025),
            random_seed=42,
            initial_capital=10000.0,
            raw_dir=Path(tmp_path / "nonexistent"),
            processed_dir=Path(settings.processed_dir),
            artifacts_dir=Path(settings.artifacts_dir),
        )
        run_research(bad_settings, data_mode="refresh")

    content_after = report_1.read_text(encoding="utf-8")
    assert content_before == content_after
