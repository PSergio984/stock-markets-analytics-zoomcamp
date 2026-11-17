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
