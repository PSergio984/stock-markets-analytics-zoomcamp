"""Verification tests for replay remediation findings.

Covers:
- Fix 1: Holdout consumption propagation across multiple research cycles, refusing to present
         subsequent research as untouched holdouts, and archiving historical releases.
- Fix 2: Strict legacy convention restriction to authenticated provenance, removal of arbitrary
         fallback, multi-metric trajectory path verification, and rejection of altered paths.
- Fix 3: Self-contained tiny authored fixtures for all critical offline unit tests, plus explicit
         opt-in marker with visible skip for actual historical binary integration tests.
- Fix 4: Compound release identity matching in dashboard, explicitly rejecting mismatched policy/data
         even with the same model hash, handling missing identities, and testing Compact/Detailed views.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app
from ai_risk_radar.contracts import DataBundle, Settings
from ai_risk_radar.pipeline import (
    _compute_sha256,
    run_final_test,
    run_research,
)
from ai_risk_radar.views import compact_view, detailed_view

# Check availability of untracked historical binary artifacts
REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_MODEL_FILE = REPO_ROOT / "artifacts" / "models" / "random_forest.joblib"
HISTORICAL_SNAPSHOT_DIR = REPO_ROOT / "data" / "raw" / "snapshot"
HISTORICAL_BINARIES_AVAILABLE = HISTORICAL_MODEL_FILE.is_file() and HISTORICAL_SNAPSHOT_DIR.is_dir()

skip_without_historical_binaries = pytest.mark.skipif(
    not HISTORICAL_BINARIES_AVAILABLE,
    reason="Opt-in historical integration test: untracked historical binary artifacts not present in clean checkout",
)


# ==============================================================================
# Self-contained Tiny Authored Fixtures (Zero Dependency on Untracked Binaries)
# ==============================================================================


def _make_tiny_authored_fixture(tmp_path: Path) -> tuple[Settings, Path]:
    """Create a fast, self-contained authored fixture with tiny synthetic series and dummy model."""
    from ai_risk_radar.config import load_settings
    from ai_risk_radar.data import save_snapshot

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
    return settings, cfg_file


# ==============================================================================
# Fix 1: Holdout Consumption Propagation & Re-Presentation Gating
# ==============================================================================


def test_fix_1_multi_cycle_holdout_consumption_propagates(tmp_path: Path):
    """Fix 1: Consumed holdout status and history propagate across multiple consecutive research cycles."""
    settings, _ = _make_tiny_authored_fixture(tmp_path)
    release_path = Path(settings.artifacts_dir) / "release.json"

    # Cycle 1: Initial research + final test evaluation
    run_research(settings, data_mode="cache")
    run_final_test(release_path, settings=settings)

    rel_c1 = json.loads(release_path.read_text(encoding="utf-8"))
    assert "test_evaluation" in rel_c1
    consumed_eval = rel_c1["test_evaluation"]

    # Cycle 2: Subsequent research cycle
    run_research(settings, data_mode="cache")
    rel_c2 = json.loads(release_path.read_text(encoding="utf-8"))
    assert rel_c2["holdout_status"] == "subsequent_research_holdout_previously_consumed"
    assert rel_c2["previous_holdout_consumption"] == consumed_eval
    assert "test_evaluation" not in rel_c2

    # Cycle 3: Third research cycle - must still preserve original holdout consumption!
    run_research(settings, data_mode="cache")
    rel_c3 = json.loads(release_path.read_text(encoding="utf-8"))
    assert rel_c3["holdout_status"] == "subsequent_research_holdout_previously_consumed"
    assert rel_c3["previous_holdout_consumption"] == consumed_eval
    assert "test_evaluation" not in rel_c3

    # Historical releases and models must be archived in artifacts/history
    history_dir = Path(settings.artifacts_dir) / "history"
    assert history_dir.is_dir()
    archived_releases = list(history_dir.glob("release_*.json"))
    assert len(archived_releases) >= 2


def test_fix_1_final_test_refuses_subsequent_research_as_untouched(tmp_path: Path):
    """Fix 1: run_final_test() strictly refuses to present subsequent research as an untouched holdout."""
    settings, _ = _make_tiny_authored_fixture(tmp_path)
    release_path = Path(settings.artifacts_dir) / "release.json"

    run_research(settings, data_mode="cache")
    run_final_test(release_path, settings=settings)

    # Trigger second research cycle
    run_research(settings, data_mode="cache")

    # Calling run_final_test on subsequent research MUST fail with ValueError before any simulation
    with pytest.raises(ValueError, match="Holdout test partition has already been consumed by a prior release"):
        run_final_test(release_path, settings=settings)


# ==============================================================================
# Fix 2: Strict Legacy Authentication & Multi-Metric Trajectory Verification
# ==============================================================================


def test_fix_2_unauthenticated_release_rejects_legacy_fallback(tmp_path: Path):
    """Fix 2: Unauthenticated release diverging from net initial capital cannot fall back to legacy."""
    settings, _ = _make_tiny_authored_fixture(tmp_path)
    release_path = Path(settings.artifacts_dir) / "release.json"

    run_research(settings, data_mode="cache")
    run_final_test(release_path, settings=settings)

    rel_data = json.loads(release_path.read_text(encoding="utf-8"))
    # Alter recorded return so net initial capital diverges, but spoof legacy return
    rel_data["test_evaluation"]["test_metrics"]["total_return"] = 0.9999
    release_path.write_text(json.dumps(rel_data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        run_final_test(release_path, settings=settings)

    # Must explicitly state net initial capital convention error and restrict legacy convention
    assert "Legacy post-funding convention is strictly restricted to authenticated historical releases" in str(
        excinfo.value
    )


def test_fix_2_multi_metric_trajectory_divergence_detected(tmp_path: Path):
    """Fix 2: Trajectory check detects altered intermediate path even if endpoint total_return matched."""
    settings, _ = _make_tiny_authored_fixture(tmp_path)
    release_path = Path(settings.artifacts_dir) / "release.json"

    run_research(settings, data_mode="cache")
    run_final_test(release_path, settings=settings)

    rel_data = json.loads(release_path.read_text(encoding="utf-8"))
    # Keep total_return identical, but tamper with max_drawdown
    orig_mdd = rel_data["test_evaluation"]["test_metrics"]["max_drawdown"]
    rel_data["test_evaluation"]["test_metrics"]["max_drawdown"] = orig_mdd - 0.10
    release_path.write_text(json.dumps(rel_data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Reproduction metric divergence on max_drawdown"):
        run_final_test(release_path, settings=settings)


# ==============================================================================
# Fix 3: Offline Self-Contained Unit Suite & Opt-In Historical Integration Test
# ==============================================================================


def test_fix_3_offline_policy_tampering_sentinel_blocks_simulation(tmp_path: Path, monkeypatch):
    """Fix 3: Policy parameter change fails BEFORE simulate() on self-contained authored fixture."""
    import ai_risk_radar.pipeline as pipeline_mod

    settings, cfg_path = _make_tiny_authored_fixture(tmp_path)
    release_path = Path(settings.artifacts_dir) / "release.json"

    run_research(settings, data_mode="cache")
    run_final_test(release_path, settings=settings)

    def sentinel_simulate(*args, **kwargs):
        raise RuntimeError("SENTINEL: simulate() was called! Check failed to block pre-simulation.")

    monkeypatch.setattr(pipeline_mod, "simulate", sentinel_simulate)

    # Tamper with cost_rate in config (0.002 instead of 0.001)
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["cost_rate"] = 0.002
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        run_final_test(release_path, config_path=cfg_path)

    assert "Policy" in str(excinfo.value) or "cost_rate" in str(excinfo.value)
    assert "SENTINEL" not in str(excinfo.value)


def test_fix_3_offline_boundary_tampering_sentinel_blocks_simulation(tmp_path: Path, monkeypatch):
    """Fix 3: Boundary change fails BEFORE simulate() on self-contained authored fixture."""
    import ai_risk_radar.pipeline as pipeline_mod

    settings, cfg_path = _make_tiny_authored_fixture(tmp_path)
    release_path = Path(settings.artifacts_dir) / "release.json"

    run_research(settings, data_mode="cache")
    run_final_test(release_path, settings=settings)

    def sentinel_simulate(*args, **kwargs):
        raise RuntimeError("SENTINEL: simulate() was called! Check failed to block pre-simulation.")

    monkeypatch.setattr(pipeline_mod, "simulate", sentinel_simulate)

    rel_data = json.loads(release_path.read_text(encoding="utf-8"))
    rel_data["boundaries"]["train_start"] = "2016-01-05"
    release_path.write_text(json.dumps(rel_data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Partition boundary mismatch on reproduction"):
        run_final_test(release_path, config_path=cfg_path)


def test_fix_3_offline_convention_tampering_sentinel_blocks_simulation(tmp_path: Path, monkeypatch):
    """Fix 3: Convention change fails BEFORE simulate() on self-contained authored fixture."""
    import ai_risk_radar.pipeline as pipeline_mod

    settings, cfg_path = _make_tiny_authored_fixture(tmp_path)
    release_path = Path(settings.artifacts_dir) / "release.json"

    run_research(settings, data_mode="cache")
    run_final_test(release_path, settings=settings)

    def sentinel_simulate(*args, **kwargs):
        raise RuntimeError("SENTINEL: simulate() was called! Check failed to block pre-simulation.")

    monkeypatch.setattr(pipeline_mod, "simulate", sentinel_simulate)

    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["target_horizon"] = 10
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Execution convention mismatch"):
        run_final_test(release_path, config_path=cfg_path)


@pytest.mark.historical_artifacts
@skip_without_historical_binaries
def test_fix_3_opt_in_actual_historical_release_reproduction(tmp_path: Path):
    """Opt-in integration test: Real historical release reproduces cleanly on isolated copies."""
    dst_artifacts = tmp_path / "artifacts"
    dst_data = tmp_path / "data" / "raw" / "snapshot"
    dst_artifacts.mkdir(parents=True, exist_ok=True)
    dst_data.mkdir(parents=True, exist_ok=True)

    shutil.copy2(REPO_ROOT / "artifacts" / "release.json", dst_artifacts / "release.json")
    shutil.copy2(REPO_ROOT / "artifacts" / "final_test_report.json", dst_artifacts / "final_test_report.json")
    (dst_artifacts / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy2(HISTORICAL_MODEL_FILE, dst_artifacts / "models" / "random_forest.joblib")

    for item in HISTORICAL_SNAPSHOT_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_data / item.name)

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

    rel_path = dst_artifacts / "release.json"
    final_test_path = dst_artifacts / "final_test_report.json"
    hash_rel_before = _compute_sha256(rel_path)
    hash_rep_before = _compute_sha256(final_test_path)

    repro_path = run_final_test(rel_path, config_path=cfg_path)
    assert repro_path.name == "reproduction_test_report.json"
    assert _compute_sha256(rel_path) == hash_rel_before
    assert _compute_sha256(final_test_path) == hash_rep_before

    repro_data = json.loads(repro_path.read_text(encoding="utf-8"))
    assert repro_data["identity"]["reproduction_status"] == "verified_match"
    assert repro_data["identity"]["metric_convention"] == "legacy_post_funding"
    assert abs(repro_data["identity"]["legacy_post_funding_total_return"] - 0.5529049846184143) == 0.0
    assert abs(repro_data["identity"]["net_initial_capital_total_return"] - 0.551353630987427) < 1e-6
    assert repro_data["identity"]["verification_scope"] == "multi_metric_trajectory_and_funding_basis"


# ==============================================================================
# Fix 4: Compound Release Identity Matching in Dashboard
# ==============================================================================


def test_fix_4_dashboard_identity_matching_cases(tmp_path: Path, monkeypatch):
    """Fix 4: Dashboard merges reproduction data ONLY on trustworthy matching compound identity."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    base_repro = {
        "identity": {
            "pipeline_mode": "final_test_reproduction",
            "reproduction_status": "verified_match",
            "release_hash": "compound_hash_abc",
            "model_hash": "model_hash_123",
            "metric_convention": "net_initial_capital",
            "net_initial_capital_total_return": 0.551,
            "legacy_post_funding_total_return": 0.553,
            "funding_cost_impact_bps": 15.5,
        },
        "allocation": {"selected_threshold": 0.50, "weights": {"SOXX": 0.35, "XLK": 0.35, "SPY": 0.2, "SHY": 0.1}},
        "data_status": {"as_of_session": "2026-11-16", "cutoff_session": "2026-11-16"},
        "forecast": {"probability": 0.25, "risk_level": "Low", "loss_definition": "def"},
        "performance": {"selected_strategy": {"total_return": 0.551, "cagr": 0.248}},
    }
    repro_file = artifacts_dir / "reproduction_test_report.json"
    repro_file.write_text(json.dumps(base_repro, indent=2), encoding="utf-8")

    latest_file = artifacts_dir / "latest_report.json"

    # Case A: Matching compound release_hash -> MERGES
    matching_latest = {
        "identity": {
            "pipeline_mode": "latest",
            "release_hash": "compound_hash_abc",
            "model_hash": "model_hash_123",
        },
        "allocation": {"selected_threshold": 0.50, "weights": {"SOXX": 0.35, "XLK": 0.35, "SPY": 0.2, "SHY": 0.1}},
        "data_status": {"as_of_session": "2026-11-16", "cutoff_session": "2026-11-16"},
        "forecast": {"probability": 0.25, "risk_level": "Low", "loss_definition": "def"},
        "performance": {},
    }
    latest_file.write_text(json.dumps(matching_latest, indent=2), encoding="utf-8")
    report, _fname = app.load_report_data()
    assert report.payload.get("reproduction_verified") is True
    assert report.payload.get("reproduction_status") == "verified_match"
    cv = compact_view(report)
    assert cv["reproduction_status"] == "verified_match"
    dv = detailed_view(report)
    assert dv["reproduction_status"] == "verified_match"

    # Case B: Differing release_hash -> REJECTED
    diff_hash_latest = dict(matching_latest)
    diff_hash_latest["identity"] = {
        "pipeline_mode": "latest",
        "release_hash": "compound_hash_DIFFERENT",
        "model_hash": "model_hash_123",
    }
    latest_file.write_text(json.dumps(diff_hash_latest, indent=2), encoding="utf-8")
    report, _ = app.load_report_data()
    assert report.payload.get("reproduction_verified") is None
    assert report.payload.get("reproduction_status") is None
    assert compact_view(report).get("reproduction_status") is None

    # Case C: Missing / Empty identity -> REJECTED
    missing_id_latest = dict(matching_latest)
    missing_id_latest["identity"] = {}
    latest_file.write_text(json.dumps(missing_id_latest, indent=2), encoding="utf-8")
    report, _ = app.load_report_data()
    assert report.payload.get("reproduction_verified") is None
    assert report.payload.get("reproduction_status") is None

    # Case D: Same model_hash, but differing policy threshold -> REJECTED
    diff_policy_latest = {
        "identity": {
            "pipeline_mode": "latest",
            "model_hash": "model_hash_123",
        },
        "allocation": {"selected_threshold": 0.35},  # Altered threshold (0.35 vs 0.50)
        "data_status": {"as_of_session": "2026-11-16", "cutoff_session": "2026-11-16"},
        "forecast": {"probability": 0.25, "risk_level": "Low", "loss_definition": "def"},
        "performance": {},
    }
    latest_file.write_text(json.dumps(diff_policy_latest, indent=2), encoding="utf-8")
    report, _ = app.load_report_data()
    assert report.payload.get("reproduction_verified") is None
    assert report.payload.get("reproduction_status") is None

    # Case E: Same model_hash, but differing data cutoff -> REJECTED
    diff_data_latest = {
        "identity": {
            "pipeline_mode": "latest",
            "model_hash": "model_hash_123",
        },
        "allocation": {"selected_threshold": 0.50},
        "data_status": {"as_of_session": "2026-11-16", "cutoff_session": "2026-12-01"},  # Altered cutoff
        "forecast": {"probability": 0.25, "risk_level": "Low", "loss_definition": "def"},
        "performance": {},
    }
    latest_file.write_text(json.dumps(diff_data_latest, indent=2), encoding="utf-8")
    report, _ = app.load_report_data()
    assert report.payload.get("reproduction_verified") is None
    assert report.payload.get("reproduction_status") is None
