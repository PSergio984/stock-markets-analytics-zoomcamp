"""Deterministic unit tests for causal feature construction and target definition."""

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.contracts import DataBundle, Settings
from ai_risk_radar.features import build_features


@pytest.fixture
def sample_settings(tmp_path):
    import json

    from ai_risk_radar.config import load_settings

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
    return load_settings(cfg_file)


def test_target_is_equal_weight_buy_and_hold(sample_settings: Settings):
    """Over 20 sessions SOXX grows 20% and XLK loses 30%; assert r20 == -0.05 and downside20 == 1."""
    sessions = pd.date_range("2020-01-01", periods=100, freq="B")

    # Base price 100 for all
    soxx_prices = [100.0] * 100
    xlk_prices = [100.0] * 100
    spy_prices = [100.0] * 100
    shy_prices = [100.0] * 100

    # At session 20, session 0 was at 100.
    # At session 20: SOXX goes to 120 (+20%), XLK goes to 70 (-30%)
    soxx_prices[20] = 120.0
    xlk_prices[20] = 70.0

    adj_close = pd.DataFrame(
        {
            "SOXX": soxx_prices,
            "XLK": xlk_prices,
            "SPY": spy_prices,
            "SHY": shy_prices,
        },
        index=sessions,
    )
    volume = pd.DataFrame(10000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    bundle = DataBundle(
        adjusted_close=adj_close,
        volume=volume,
        sessions=sessions,
        manifest={"source": "test"},
    )

    feat_set = build_features(bundle, sample_settings)
    df = feat_set.frame

    # On session 0:
    # r20 should be 0.5 * (120/100 - 1) + 0.5 * (70/100 - 1) = 0.5 * 0.20 + 0.5 * (-0.30) = -0.05
    r20_session_0 = df.loc[sessions[0], "r20"]
    downside20_session_0 = df.loc[sessions[0], "downside20"]

    assert r20_session_0 == pytest.approx(-0.05, abs=1e-6)
    assert downside20_session_0 == 1
    assert df.loc[sessions[0], "label_end"] == sessions[20]


def test_unknown_tail_labels_preserve_inference_rows(sample_settings: Settings):
    """Assert final 20 sessions have unknown outcomes (NaN) but complete trailing features."""
    sessions = pd.date_range("2020-01-01", periods=120, freq="B")
    adj_close = pd.DataFrame(
        {col: 100.0 + np.arange(120) * (i + 1) for i, col in enumerate(["SOXX", "XLK", "SPY", "SHY"])},
        index=sessions,
    )
    volume = pd.DataFrame(10000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    bundle = DataBundle(
        adjusted_close=adj_close,
        volume=volume,
        sessions=sessions,
        manifest={"source": "test"},
    )

    feat_set = build_features(bundle, sample_settings)
    df = feat_set.frame

    # Last 20 rows must have NaN target labels
    tail_20 = df.iloc[-20:]
    assert tail_20["r20"].isna().all()
    assert tail_20["downside20"].isna().all()

    # But is_feature_complete should be True since index > 63 warmup
    assert tail_20["is_feature_complete"].all()


def test_future_price_change_cannot_change_past_features(sample_settings: Settings):
    """Assert changing prices at t=80 cannot alter features at t=40 (strict causality)."""
    sessions = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices_base = pd.DataFrame(
        100.0 + np.cumsum(np.random.randn(100, 4), axis=0),
        index=sessions,
        columns=["SOXX", "XLK", "SPY", "SHY"],
    )
    volume = pd.DataFrame(10000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])

    bundle_base = DataBundle(
        adjusted_close=prices_base,
        volume=volume,
        sessions=sessions,
        manifest={},
    )
    feat_base = build_features(bundle_base, sample_settings)

    # Modify prices at t=80 onwards
    prices_mod = prices_base.copy()
    prices_mod.iloc[80:, 0] *= 2.0  # Double SOXX after t=80

    bundle_mod = DataBundle(
        adjusted_close=prices_mod,
        volume=volume,
        sessions=sessions,
        manifest={},
    )
    feat_mod = build_features(bundle_mod, sample_settings)

    # Feature columns up to session 79 must be identical
    feature_cols = list(feat_base.feature_columns)
    pd.testing.assert_frame_equal(
        feat_base.frame.loc[: sessions[79], feature_cols],
        feat_mod.frame.loc[: sessions[79], feature_cols],
    )
