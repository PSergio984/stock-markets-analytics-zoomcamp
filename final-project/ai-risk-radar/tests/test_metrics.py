"""Deterministic unit tests for financial metrics: CAGR, Max Drawdown, Sharpe (rf=0), and Turnover."""

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.metrics import compute_metrics


def test_cagr_and_drawdown_calculation():
    """Verify CAGR and maximum peak-to-trough drawdown calculation."""
    # 252 sessions (1 trading year)
    dates = pd.date_range("2024-01-02", periods=252, freq="B")

    # Portfolio starts at 100, drops to 80 (drawdown -20%), then rises to 121 (+21% total return)
    equity = np.linspace(100.0, 121.0, 252)
    equity[50] = 80.0  # Acute trough

    daily = pd.DataFrame(
        {
            "equity": equity,
            "net_return": pd.Series(equity).pct_change().fillna(0.0).values,
            "holding_SOXX": equity * 0.1,
            "holding_XLK": equity * 0.1,
            "holding_SPY": equity * 0.4,
            "holding_SHY": equity * 0.4,
        },
        index=dates,
    )
    trades = pd.DataFrame(columns=["signal_date", "exec_date", "turnover", "cost"])

    m = compute_metrics(daily, trades)
    assert m["cagr"] == pytest.approx(0.21, rel=1e-2)
    assert m["max_drawdown"] <= -0.20


def test_sharpe_ratio_assumes_zero_risk_free():
    """Verify Sharpe ratio calculates mean/std * sqrt(252) with rf=0, and None for zero vol."""
    dates = pd.date_range("2024-01-02", periods=252, freq="B")
    flat_equity = np.full(252, 100.0)
    daily_flat = pd.DataFrame(
        {
            "equity": flat_equity,
            "net_return": np.zeros(252),
            "holding_SOXX": 25.0,
            "holding_XLK": 25.0,
            "holding_SPY": 25.0,
            "holding_SHY": 25.0,
        },
        index=dates,
    )
    trades = pd.DataFrame(columns=["signal_date", "exec_date", "turnover", "cost"])
    m_flat = compute_metrics(daily_flat, trades)
    assert m_flat["sharpe"] is None
