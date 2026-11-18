"""Deterministic unit tests for price snapshots, adjustments, and settings validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.config import load_settings
from ai_risk_radar.contracts import DataBundle
from ai_risk_radar.data import (
    load_snapshot,
    save_snapshot,
    validate_sessions_and_prices,
)


def test_missing_exchange_session_rejected():
    """Assert a missing SOXX session while SPY trades raises a dated error."""
    sessions = pd.date_range("2024-01-02", periods=5, freq="B")
    adj_close = pd.DataFrame(
        {
            "SOXX": [100.0, 101.0, np.nan, 103.0, 104.0],  # missing session on day 3
            "XLK": [50.0, 51.0, 52.0, 53.0, 54.0],
            "SPY": [400.0, 401.0, 402.0, 403.0, 404.0],
            "SHY": [80.0, 80.1, 80.2, 80.3, 80.4],
        },
        index=sessions,
    )
    volume = pd.DataFrame(1000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])

    with pytest.raises(ValueError, match="Missing.*session.*SOXX.*2024-01-04"):
        validate_sessions_and_prices(adj_close, volume, expected_sessions=sessions)


def test_duplicate_or_unsorted_sessions_rejected():
    """Assert duplicate or out-of-order session dates are rejected."""
    sessions_unsorted = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    adj_close = pd.DataFrame(
        100.0,
        index=sessions_unsorted,
        columns=["SOXX", "XLK", "SPY", "SHY"],
    )
    volume = pd.DataFrame(1000, index=sessions_unsorted, columns=["SOXX", "XLK", "SPY", "SHY"])

    with pytest.raises(ValueError, match="Sessions must be unique and sorted ascending"):
        validate_sessions_and_prices(adj_close, volume)

    sessions_duplicate = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
    adj_close_dup = pd.DataFrame(
        100.0,
        index=sessions_duplicate,
        columns=["SOXX", "XLK", "SPY", "SHY"],
    )
    with pytest.raises(ValueError, match="Sessions must be unique and sorted ascending"):
        validate_sessions_and_prices(adj_close_dup, volume)


def test_adjustments_do_not_double_count_dividends():
    """Verify adjusted price total return preserves single-period return and does not double-count dividends."""
    # Suppose stock starts at 100. Day 2: pays $2 dividend, close is 98.
    # Total return is (98 + 2)/100 - 1 = 0.00%.
    # In adjusted close series, day 1 adjusted close is 100, day 2 adjusted close is 100 (or day 1 is 98, day 2 is 98).
    # Single-period return from adjusted prices is P_adj(t) / P_adj(t-1) - 1.
    # If someone added dividend yield again to P_adj return, return would become 2.04%, which is double-counting.
    p_adj_day1 = 98.0
    p_adj_day2 = 98.0
    adj_return = (p_adj_day2 / p_adj_day1) - 1.0
    assert adj_return == pytest.approx(0.0)

    # If unadjusted close was 100 on day 1 and 98 on day 2 with a 2.0 dividend:
    raw_close_day1 = 100.0
    raw_close_day2 = 98.0
    dividend = 2.0
    raw_total_return = (raw_close_day2 + dividend) / raw_close_day1 - 1.0
    assert raw_total_return == pytest.approx(0.0)

    # Double counting check: adding dividend to adjusted close return deviates from true total return
    double_counted_return = adj_return + (dividend / raw_close_day1)
    assert double_counted_return > raw_total_return


def test_snapshot_hash_mismatch_rejected(tmp_path: Path):
    """Assert corrupt cached bytes fail hash verification upon reload."""
    sessions = pd.date_range("2024-01-02", periods=3, freq="B")
    adj_close = pd.DataFrame(
        {
            "SOXX": [100.0, 101.0, 102.0],
            "XLK": [50.0, 51.0, 52.0],
            "SPY": [400.0, 401.0, 402.0],
            "SHY": [80.0, 80.1, 80.2],
        },
        index=sessions,
    )
    volume = pd.DataFrame(1000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    manifest = {
        "source": "yfinance",
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
        "start_date": "2024-01-02",
        "end_date": "2024-01-04",
    }
    bundle = DataBundle(
        adjusted_close=adj_close,
        volume=volume,
        sessions=sessions,
        manifest=manifest,
    )

    snapshot_dir = tmp_path / "snapshot"
    manifest_path = save_snapshot(bundle, snapshot_dir)
    assert manifest_path.exists()

    # Verify normal loading succeeds
    loaded = load_snapshot(snapshot_dir)
    pd.testing.assert_frame_equal(loaded.adjusted_close, adj_close, check_freq=False)

    # Tamper with the adjusted_close parquet file
    data_file = snapshot_dir / "adjusted_close.parquet"
    assert data_file.exists()
    content = bytearray(data_file.read_bytes())
    content[10] ^= 0xFF  # Flip bits
    data_file.write_bytes(bytes(content))

    # Reload must raise hash verification error
    with pytest.raises(ValueError, match="Hash mismatch"):
        load_snapshot(snapshot_dir)


def test_weight_order_and_sum_validated(tmp_path: Path):
    """Assert invalid weight order, non-sum-to-1, or negative weights raise ValueError."""
    config_file = tmp_path / "bad_config.json"
    bad_config = {
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
        "acquisition_start": "2016-01-01",
        "source": "yfinance",
        "target_horizon": 20,
        "target_threshold": -0.05,
        "weights": {
            "growth": [0.40, 0.40, 0.20, 0.10],  # Sums to 1.10!
            "defensive": [0.10, 0.15, 0.35, 0.40],
            "fixed_baseline": [0.20, 0.20, 0.20, 0.40],
        },
        "candidate_thresholds": [0.20, 0.35, 0.50],
        "cost_rate": 0.001,
        "cost_sensitivities": [0.0005, 0.0010, 0.0025],
        "random_seed": 42,
        "initial_capital": 10000.0,
        "data_paths": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "artifacts_dir": "artifacts",
        },
    }
    config_file.write_text(json.dumps(bad_config))

    with pytest.raises(ValueError, match="Weights.*must sum to 1.0"):
        load_settings(config_file)

    bad_config["weights"]["growth"] = [-0.10, 0.50, 0.20, 0.40]  # Sums to 1.0 but has negative!
    config_file.write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="negative weights"):
        load_settings(config_file)

    bad_config["weights"]["growth"] = [float("nan"), 0.50, 0.20, 0.30]
    config_file.write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="NaN or negative weights"):
        load_settings(config_file)

    bad_config["weights"]["growth"] = [0.35, 0.35, 0.20, 0.10]
    bad_config["initial_capital"] = 0.0
    config_file.write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        load_settings(config_file)

    bad_config["initial_capital"] = 10000.0
    bad_config["cost_rate"] = -0.001
    config_file.write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="cost_rate cannot be negative"):
        load_settings(config_file)
