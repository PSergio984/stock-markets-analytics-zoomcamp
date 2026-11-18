"""Verification tests for seven review findings from replay-agy.md.

Covers:
- Finding 1: Actual historical release reproduction on legacy_post_funding metric convention.
- Finding 2: Full reproduction test on isolated copy of actual historical release artifacts.
- Finding 3: Pre-simulation sentinel proof: policy/config parameter changes fail before simulate().
- Finding 4: Pre-simulation boundary validation: all 9 partition dates & conventions checked before simulate().
- Finding 5: Holdout consumption preservation across research releases.
- Finding 6: Dashboard data loading & views support reproduction status and dual-convention returns.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.contracts import DataBundle
from ai_risk_radar.pipeline import (
    _compute_sha256,
    run_final_test,
    run_research,
)
from ai_risk_radar.views import compact_view, detailed_view


def _copy_historical_project(tmp_path: Path) -> tuple[Path, Path]:
    """Copy actual project artifacts to an isolated tmp_path."""
    root_dir = Path(__file__).resolve().parent.parent

    # Destination paths
    dst_artifacts = tmp_path / "artifacts"
    dst_data = tmp_path / "data" / "raw" / "snapshot"
    dst_artifacts.mkdir(parents=True, exist_ok=True)
    dst_data.mkdir(parents=True, exist_ok=True)

    # Copy artifacts
    shutil.copy2(root_dir / "artifacts" / "release.json", dst_artifacts / "release.json")
    shutil.copy2(root_dir / "artifacts" / "final_test_report.json", dst_artifacts / "final_test_report.json")
    (dst_artifacts / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        root_dir / "artifacts" / "models" / "random_forest.joblib",
        dst_artifacts / "models" / "random_forest.joblib",
    )

    # Copy snapshot
    src_snapshot = root_dir / "data" / "raw" / "snapshot"
    for item in src_snapshot.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_data / item.name)

    # Write isolated config.json
    cfg = {
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
        "acquisition_start": "2016-01-04",
        "source": "toy",
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
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "artifacts_dir": str(dst_artifacts),
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return dst_artifacts / "release.json", cfg_path


def test_finding_1_and_2_actual_historical_release_reproduction(tmp_path: Path):
    """Findings 1 & 2: Real historical release reproduces cleanly with bit-exact post-funding return."""
    rel_path, cfg_path = _copy_historical_project(tmp_path)
    final_test_path = rel_path.parent / "final_test_report.json"

    # Baseline SHA-256 digests before reproduction
    hash_rel_before = _compute_sha256(rel_path)
    hash_rep_before = _compute_sha256(final_test_path)
    assert hash_rel_before == "1f4751a7108874364533b55f38a5ab1fdd3e26b0689dfbd502319e29098319b9"
    assert hash_rep_before == "6603ab4c7204bd38b2d179ee67bf17760788240e11f4fc61b267d3c3705d0901"

    # Execute reproduction
    repro_path = run_final_test(rel_path, config_path=cfg_path)
    assert repro_path.name == "reproduction_test_report.json"
    assert repro_path.is_file()

    # Verify original artifacts remain 100% byte-for-byte immutable
    assert _compute_sha256(rel_path) == hash_rel_before
    assert _compute_sha256(final_test_path) == hash_rep_before

    # Verify reproduction report content
    repro_data = json.loads(repro_path.read_text(encoding="utf-8"))
    assert repro_data["identity"]["pipeline_mode"] == "final_test_reproduction"
    assert repro_data["identity"]["reproduction_status"] == "verified_match"
    assert repro_data["identity"]["metric_convention"] == "legacy_post_funding"

    # Bit-exact legacy trajectory return check (diff == 0.0)
    orig_total_return = 0.5529049846184143
    legacy_return = repro_data["identity"]["legacy_post_funding_total_return"]
    assert abs(legacy_return - orig_total_return) == 0.0

    # Net initial capital return (~55.14%)
    net_return = repro_data["identity"]["net_initial_capital_total_return"]
    assert abs(net_return - 0.551353630987427) < 1e-6

    # Funding impact (~15.51 bps)
    funding_bps = repro_data["identity"]["funding_cost_impact_bps"]
    assert abs(funding_bps - 15.513536) < 1e-3


def test_finding_3_pre_simulation_sentinel_policy_tampering(tmp_path: Path, monkeypatch):
    """Finding 3: Policy parameter change on historical release fails BEFORE simulate() is called."""
    import ai_risk_radar.pipeline as pipeline_mod

    rel_path, cfg_path = _copy_historical_project(tmp_path)

    def sentinel_simulate(*args, **kwargs):
        raise RuntimeError("SENTINEL: simulate() was called! Check failed to block pre-simulation.")

    monkeypatch.setattr(pipeline_mod, "simulate", sentinel_simulate)

    # Tamper with cost_rate in config (0.002 instead of 0.001)
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["cost_rate"] = 0.002
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        run_final_test(rel_path, config_path=cfg_path)

    # Must raise ValueError for policy mismatch, NOT the RuntimeError sentinel
    assert "Policy" in str(excinfo.value) or "cost_rate" in str(excinfo.value)
    assert "SENTINEL" not in str(excinfo.value)


def test_finding_4_pre_simulation_sentinel_boundary_tampering(tmp_path: Path, monkeypatch):
    """Finding 4: Boundary partition changes fail BEFORE simulate() is called."""
    import ai_risk_radar.pipeline as pipeline_mod

    rel_path, cfg_path = _copy_historical_project(tmp_path)

    def sentinel_simulate(*args, **kwargs):
        raise RuntimeError("SENTINEL: simulate() was called! Check failed to block pre-simulation.")

    monkeypatch.setattr(pipeline_mod, "simulate", sentinel_simulate)

    # Tamper with train_start boundary in release.json
    rel_data = json.loads(rel_path.read_text(encoding="utf-8"))
    rel_data["boundaries"]["train_start"] = "2016-01-05"
    rel_path.write_text(json.dumps(rel_data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Partition boundary mismatch on reproduction"):
        run_final_test(rel_path, config_path=cfg_path)


def test_finding_4_pre_simulation_convention_tampering(tmp_path: Path, monkeypatch):
    """Finding 4: Convention changes (target_horizon) fail BEFORE simulate() is called."""
    import ai_risk_radar.pipeline as pipeline_mod

    rel_path, cfg_path = _copy_historical_project(tmp_path)

    def sentinel_simulate(*args, **kwargs):
        raise RuntimeError("SENTINEL: simulate() was called! Check failed to block pre-simulation.")

    monkeypatch.setattr(pipeline_mod, "simulate", sentinel_simulate)

    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["target_horizon"] = 10
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Execution convention mismatch"):
        run_final_test(rel_path, config_path=cfg_path)


def test_finding_5_holdout_consumption_preservation_across_research(tmp_path: Path):
    """Finding 5: run_research() preserves previous holdout consumption history."""
    from ai_risk_radar.config import load_settings
    from ai_risk_radar.data import save_snapshot

    cfg = {
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
        "acquisition_start": "2020-01-01",
        "source": "toy",
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
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    settings = load_settings(cfg_file)

    sessions = pd.date_range("2016-01-04", "2026-11-16", freq="B")
    n = len(sessions)
    np.random.seed(42)
    prices = pd.DataFrame(
        {
            "SOXX": 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
            "XLK": 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
            "SPY": 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.005)),
            "SHY": 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.0005)),
        },
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

    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    release_path = artifacts_dir / "release.json"

    # Pre-existing release that had consumed holdout
    fake_eval = {
        "evaluated_at": "2026-11-17T12:00:00",
        "test_metrics": {"total_return": 0.50},
    }
    release_path.write_text(json.dumps({"test_evaluation": fake_eval}), encoding="utf-8")

    # Run research
    run_research(settings, data_mode="cache")

    # Read back release.json
    new_release_data = json.loads(release_path.read_text(encoding="utf-8"))
    assert new_release_data["holdout_status"] == "subsequent_research_holdout_previously_consumed"
    assert new_release_data["previous_holdout_consumption"] == fake_eval


def test_finding_6_dashboard_loads_reproduction_and_views_format_dual_returns(tmp_path: Path, monkeypatch):
    """Finding 6: load_report_data enriches reproduction and views expose dual returns."""
    import app

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Mock working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    repro_payload = {
        "identity": {
            "name": "AI Investment Risk Radar",
            "pipeline_mode": "final_test_reproduction",
            "reproduction_status": "verified_match",
            "metric_convention": "legacy_post_funding",
            "legacy_post_funding_total_return": 0.55290498,
            "net_initial_capital_total_return": 0.55135363,
            "funding_cost_impact_bps": 15.51,
        },
        "performance": {
            "evaluation_partition": "test",
            "selected_strategy": {"total_return": 0.55135363, "cagr": 0.248},
            "metric_convention": "legacy_post_funding",
            "legacy_post_funding_total_return": 0.55290498,
            "net_initial_capital_total_return": 0.55135363,
            "funding_cost_impact_bps": 15.51,
        },
        "forecast": {
            "probability": 0.30,
            "probability_status": "calibrated",
            "risk_level": "Medium",
        },
        "allocation": {
            "strategy_mode": "Growth",
            "selected_threshold": 0.50,
            "weights": {"SOXX": 0.35, "XLK": 0.35, "SPY": 0.20, "SHY": 0.10},
        },
        "data_status": {"as_of_session": "2026-11-16"},
    }

    repro_file = artifacts_dir / "reproduction_test_report.json"
    repro_file.write_text(json.dumps(repro_payload, indent=2), encoding="utf-8")

    report, fname = app.load_report_data()
    assert report is not None
    assert fname == "reproduction_test_report.json"

    c_v = compact_view(report)
    assert c_v["reproduction_status"] == "verified_match"
    assert c_v["legacy_post_funding_total_return"] == 0.55290498
    assert c_v["net_initial_capital_total_return"] == 0.55135363
    assert c_v["funding_cost_impact_bps"] == 15.51

    d_v = detailed_view(report)
    assert d_v["reproduction_status"] == "verified_match"
