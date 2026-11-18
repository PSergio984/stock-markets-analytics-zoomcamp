"""Deterministic unit tests for trade execution delay, drifted holdings, and transaction costs."""

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.contracts import BacktestResult, DataBundle, Settings
from ai_risk_radar.simulation import (
    select_strategy,
    simulate,
    weekly_targets,
)


@pytest.fixture
def test_settings(tmp_path):
    import json

    from ai_risk_radar.config import load_settings

    cfg = {
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
        "acquisition_start": "2024-01-01",
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


def test_initial_funding_cost_is_charged(test_settings: Settings):
    """Initial all-cash funding charges cost c such that net wealth factor is 1 / (1 + c)."""
    # 2 flat sessions, constant prices
    sessions = pd.date_range("2024-01-02", periods=2, freq="B")
    prices = pd.DataFrame(100.0, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    targets = pd.DataFrame(
        [test_settings.weights_growth, test_settings.weights_growth],
        index=sessions,
        columns=["SOXX", "XLK", "SPY", "SHY"],
    )

    c = 0.001
    res = simulate(prices, targets, test_settings, cost_rate=c)
    expected_initial_equity = test_settings.initial_capital / (1.0 + c)

    assert res.daily.loc[sessions[0], "equity"] == pytest.approx(expected_initial_equity, rel=1e-5)


def test_costs_charge_each_buy_and_sell(test_settings: Settings):
    """For a full SOXX-to-SHY transfer with zero market return at rate c, net wealth factor is (1-c)/(1+c)."""
    sessions = pd.date_range("2024-01-02", periods=2, freq="B")
    prices = pd.DataFrame(100.0, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])

    # Day 0: 100% SOXX. Day 1: 100% SHY
    targets = pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        index=sessions,
        columns=["SOXX", "XLK", "SPY", "SHY"],
    )

    c = 0.001
    res = simulate(prices, targets, test_settings, cost_rate=c)

    # Initial funding: V0 / (1+c)
    # Day 1 transfer from SOXX to SHY: Vpost = Vpre * (1-c)/(1+c)
    # Total wealth factor after transfer: (V0 / (1+c)) * ((1-c) / (1+c))
    v0 = test_settings.initial_capital
    v_day0 = v0 / (1.0 + c)
    v_day1 = v_day0 * (1.0 - c) / (1.0 + c)

    assert res.daily.loc[sessions[1], "equity"] == pytest.approx(v_day1, rel=1e-5)


def test_buy_and_hold_weights_drift(test_settings: Settings):
    """Weights drift when price returns differ and no rebalance occurs."""
    sessions = pd.date_range("2024-01-02", periods=3, freq="B")
    # SOXX doubles on day 1, others stay flat
    prices = pd.DataFrame(
        {
            "SOXX": [100.0, 200.0, 200.0],
            "XLK": [100.0, 100.0, 100.0],
            "SPY": [100.0, 100.0, 100.0],
            "SHY": [100.0, 100.0, 100.0],
        },
        index=sessions,
    )
    # Target only set on day 0 (buy and hold)
    targets = pd.DataFrame(
        [
            [0.25, 0.25, 0.25, 0.25],
            [np.nan, np.nan, np.nan, np.nan],  # No trade on day 1
            [np.nan, np.nan, np.nan, np.nan],  # No trade on day 2
        ],
        index=sessions,
        columns=["SOXX", "XLK", "SPY", "SHY"],
    )

    res = simulate(prices, targets, test_settings, cost_rate=0.0)

    # On day 1, SOXX doubled from 25 to 50, total portfolio is 50+25+25+25 = 125
    # SOXX weight drifted to 50 / 125 = 0.40 (40%)
    holdings = res.daily.loc[sessions[1]]
    soxx_weight = holdings["holding_SOXX"] / holdings["equity"]
    assert soxx_weight == pytest.approx(0.40, rel=1e-4)


def test_holiday_week_executes_next_session_close(test_settings: Settings):
    """Holiday-shortened week signals on actual last session, executes on next trading session close."""
    # Week with Monday Jan 15 holiday (MLK Day): sessions are Fri Jan 12, Tue Jan 16
    sessions = pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17"])
    prices = pd.DataFrame(100.0, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    volume = pd.DataFrame(1000, index=sessions, columns=["SOXX", "XLK", "SPY", "SHY"])
    bundle = DataBundle(
        adjusted_close=prices,
        volume=volume,
        sessions=sessions,
        manifest={},
    )

    # Risk signals high (>0.35) on Fri Jan 12
    risk_series = pd.Series(
        [0.10, 0.10, 0.60, 0.10, 0.10],
        index=sessions,
    )

    targets = weekly_targets(risk_series, bundle, test_settings, threshold=0.35)

    # Signal on Friday Jan 12 must trigger execution on Tuesday Jan 16 (the next trading session)
    # Monday was holiday so no session exists on Jan 15
    assert not targets.loc["2024-01-16"].isna().all()
    # Friday Jan 12 itself should NOT execute the new weights
    assert targets.loc["2024-01-12"].isna().all()


def test_no_qualifying_candidate_returns_none(test_settings: Settings):
    """Assert select_strategy returns None if no candidate achieves CAGR >= fixed baseline."""
    baseline = BacktestResult(
        daily=pd.DataFrame(),
        trades=pd.DataFrame(),
        metrics={"cagr": 0.10, "max_drawdown": -0.15, "turnover": 0.5},
    )
    # Candidates with CAGR lower than baseline (0.05 < 0.10)
    candidates = {
        "th_0.20": BacktestResult(
            daily=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={"cagr": 0.05, "max_drawdown": -0.08, "turnover": 0.8},
        ),
        "th_0.35": BacktestResult(
            daily=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={"cagr": 0.07, "max_drawdown": -0.10, "turnover": 0.6},
        ),
    }

    selected = select_strategy(candidates, baseline)
    assert selected is None
