"""RED/GREEN verification tests for Independent Review Round 1 findings."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.contracts import DataBundle, Settings
from ai_risk_radar.reporting import _clean_json_value
from ai_risk_radar.simulation import benchmark_targets, simulate


@pytest.fixture
def sample_settings(tmp_path: Path) -> Settings:
    return Settings(
        tickers=("SOXX", "XLK", "SPY", "SHY"),
        acquisition_start="2024-01-01",
        source="toy",
        target_horizon=20,
        target_threshold=-0.05,
        weights_growth=(0.35, 0.35, 0.20, 0.10),
        weights_defensive=(0.10, 0.15, 0.35, 0.40),
        weights_fixed_baseline=(0.20, 0.20, 0.20, 0.40),
        candidate_thresholds=(0.20, 0.35, 0.50),
        cost_rate=0.001,  # 10 bps
        cost_sensitivities=(0.0005, 0.0010, 0.0025),
        random_seed=42,
        initial_capital=10000.0,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_agy_1_benchmark_targets_default_sessions_none(sample_settings: Settings):
    """AGY-1: benchmark_targets must not crash with TypeError when sessions=None."""
    sessions = pd.date_range("2024-01-01", periods=70, freq="B")
    prices = pd.DataFrame(100.0, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    volume = pd.DataFrame(1000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    bundle = DataBundle(adjusted_close=prices, volume=volume, sessions=sessions, manifest={})

    # This raised TypeError: '>' not supported between instances of 'NoneType' and 'Timestamp'
    targets = benchmark_targets(bundle, sample_settings, sessions=None)
    assert "growth_buy_and_hold" in targets
    assert "fixed_defensive_weekly" in targets
    assert "momentum_63" in targets


def test_agy_3_metrics_accounts_for_initial_funding_cost(sample_settings: Settings):
    """AGY-3: Flat market must reflect initial funding transaction cost in total return and CAGR."""
    sessions = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(100.0, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    targets = pd.DataFrame(np.nan, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    targets.iloc[0] = [0.35, 0.35, 0.20, 0.10]

    res = simulate(prices, targets, sample_settings, cost_rate=0.001)

    # In a 0% return market, starting capital $10,000 pays 10 bps funding = $9,990.01 post-trade.
    # Total return on invested capital must be -0.0999% (-0.000999), NOT 0.000000.
    expected_ret = (res.daily["equity"].iloc[-1] - sample_settings.initial_capital) / sample_settings.initial_capital
    assert abs(res.metrics["total_return"] - expected_ret) < 1e-6
    assert res.metrics["total_return"] < 0.0  # Must be negative due to friction


def test_agy_5_clean_json_value_preserves_boolean_types():
    """AGY-5: _clean_json_value must preserve True/False as bool, not integer 1/0."""
    cleaned_true = _clean_json_value(True)
    cleaned_false = _clean_json_value(False)

    assert isinstance(cleaned_true, bool)
    assert type(cleaned_true) is bool
    assert cleaned_true is True

    assert isinstance(cleaned_false, bool)
    assert type(cleaned_false) is bool
    assert cleaned_false is False


def test_agy_11_consumed_holdout_blocks_silent_reuse(tmp_path: Path):
    """AGY-11: Attempting to run final-test on an already-consumed holdout must raise ValueError."""
    import json

    from ai_risk_radar.pipeline import run_final_test

    release_file = tmp_path / "release.json"
    release_data = {
        "release_version": "1.0.0",
        "hashes": {"model": "dummy", "data": "dummy"},
        "test_evaluation": {"evaluated_at": "2026-11-17T00:00:00", "test_metrics": {}},
    }
    release_file.write_text(json.dumps(release_data), encoding="utf-8")

    with pytest.raises(ValueError, match="consumed"):
        run_final_test(release_file)


def test_agy_11_policy_tampering_blocks_final_test(tmp_path: Path):
    """AGY-11: Policy parameter changes after research must fail policy hash verification."""
    import json

    from ai_risk_radar.contracts import Settings
    from ai_risk_radar.pipeline import _compute_policy_hash, run_final_test

    settings = Settings(
        tickers=("SOXX", "XLK", "SPY", "SHY"),
        acquisition_start="2024-01-01",
        source="toy",
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
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )
    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True)
    model_file = models_dir / "random_forest.joblib"
    model_file.write_bytes(b"dummy_weights")

    import hashlib
    model_hash = hashlib.sha256(b"dummy_weights").hexdigest()
    orig_policy_hash = _compute_policy_hash(0.50, settings)

    release_file = tmp_path / "artifacts" / "release.json"
    release_data = {
        "release_version": "1.0.0",
        "hashes": {"model": model_hash, "data": "unknown", "policy": orig_policy_hash},
        "selected_threshold": 0.35,  # Tampered: policy does not match orig_policy_hash
    }
    release_file.write_text(json.dumps(release_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Policy / configuration mismatch"):
        run_final_test(release_file, settings=settings)


def test_agy_11_identity_distinct_model_and_release_hashes():
    """AGY-11: identity.model_hash and identity.release_hash must be distinct and non-misleading."""
    from ai_risk_radar.contracts import Settings
    from ai_risk_radar.pipeline import _compute_policy_hash, _compute_release_hash

    settings = Settings(
        tickers=("SOXX", "XLK", "SPY", "SHY"),
        acquisition_start="2024-01-01",
        source="toy",
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
        raw_dir=Path("raw"),
        processed_dir=Path("processed"),
        artifacts_dir=Path("artifacts"),
    )

    model_hash = "a" * 64
    data_hash = "b" * 64
    policy_hash = _compute_policy_hash(0.50, settings)
    compound_release_hash = _compute_release_hash(model_hash, data_hash, policy_hash)

    # Release hash must combine model + data + policy and cannot equal model_hash alone
    assert compound_release_hash != model_hash
    assert len(compound_release_hash) == 64
    assert len(model_hash) == 64

