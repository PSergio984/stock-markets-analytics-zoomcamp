"""Deterministic unit tests for Compact and Detailed dashboard view formatters."""

import pytest

from ai_risk_radar.contracts import Report
from ai_risk_radar.views import compact_view, detailed_view, format_percentage_points


@pytest.fixture
def sample_report():
    payload = {
        "identity": {
            "name": "AI Investment Risk Radar",
            "version": "0.1.0",
            "pipeline_mode": "final_test",
            "release_hash": "abc123hash",
        },
        "data_status": {
            "source": "yfinance",
            "as_of_session": "2026-11-16",
            "staleness_sessions": 2,
            "refresh_status": "ok",
        },
        "forecast": {
            "as_of_session": "2026-11-16",
            "horizon_sessions": 20,
            "loss_threshold": -0.05,
            "loss_definition": "Estimated probability of reference basket total return <= -5.0% over next 20 trading sessions",
            "probability": 0.454,
            "probability_status": "calibrated",
            "fallback_reason": None,
            "raw_probability": 0.454,
            "risk_level": "High",
            "feature_explanation": ["Drawdown 63 measured", "Vol 20 measured"],
        },
        "allocation": {
            "strategy_mode": "Growth",
            "selected_threshold": 0.50,
            "weights": {"SOXX": 0.35, "XLK": 0.35, "SPY": 0.20, "SHY": 0.10},
            "trade_required": True,
            "trade_description": "Maintain growth exposure",
        },
        "performance": {
            "evaluation_partition": "test",
            "selected_strategy": {
                "cagr": 0.248,
                "max_drawdown": -0.248,
                "sharpe": 0.93,
                "total_return": 0.552,
                "turnover": 2.15,
                "defensive_occupancy": 0.05,
            },
        },
        "comparisons": {
            "growth_buy_and_hold": {
                "cagr": 0.257,
                "max_drawdown": -0.256,
                "sharpe": 0.91,
                "total_return": 0.573,
                "turnover": 0.50,
            },
            "fixed_defensive_weekly": {
                "cagr": 0.172,
                "max_drawdown": -0.158,
                "sharpe": 1.01,
                "total_return": 0.369,
                "turnover": 1.35,
            },
        },
        "diagnostics": {
            "brier_score": 0.18,
            "reliability_bins": [{"bin": "0.4-0.6", "count": 25, "obs_mean": 0.48}],
        },
        "limitations": ["Proxy limitations", "Zero risk-free Sharpe assumption"],
    }
    return Report(payload=payload)


def test_modes_share_forecast_and_metrics(sample_report: Report):
    """Assert Compact and Detailed views share exact identical forecast and performance metrics."""
    compact = compact_view(sample_report)
    detailed = detailed_view(sample_report)

    assert compact["probability"] == detailed["probability"]
    assert compact["risk_level"] == detailed["risk_level"]
    assert compact["selected_metrics"] == detailed["selected_metrics"]
    assert compact["allocation"] == detailed["allocation"]


def test_missing_or_stale_report_is_visible(sample_report: Report):
    """Assert stale report (>0 staleness sessions) produces a visible warning indicator."""
    compact = compact_view(sample_report)
    assert compact["is_stale"] is True
    assert compact["staleness_sessions"] == 2


def test_forecast_label_states_20_sessions_and_5_percent(sample_report: Report):
    """Assert event description explicitly mentions 20 trading sessions and 5% threshold."""
    compact = compact_view(sample_report)
    desc = compact["event_definition"]
    assert "20" in desc
    assert "5" in desc


def test_missing_calibration_never_shows_probability():
    """Assert unavailable calibration status displays probability as None / 'Unavailable'."""
    payload = {
        "identity": {},
        "data_status": {"as_of_session": "2026-11-16", "staleness_sessions": 0},
        "forecast": {
            "probability": None,
            "probability_status": "unavailable",
            "fallback_reason": "Inadequate positive events in calibration",
            "loss_definition": "Loss <= -5% over 20 sessions",
            "feature_explanation": [],
        },
        "allocation": {
            "strategy_mode": "Fixed Baseline (Fallback)",
            "weights": {"SOXX": 0.2, "XLK": 0.2, "SPY": 0.2, "SHY": 0.4},
            "trade_description": "Fallback to fixed baseline",
        },
        "performance": {"selected_strategy": {}},
        "comparisons": {},
        "diagnostics": {},
        "limitations": [],
    }
    rep = Report(payload=payload)
    compact = compact_view(rep)
    assert compact["probability"] is None
    assert compact["probability_display"] == "Unavailable"


def test_percentage_points_are_not_relative_percentages():
    """Assert total return differences format as percentage points (pp), not relative percentages."""
    # E.g. 25% return vs 20% return is +5.0 percentage points (pp), NOT +25%
    diff_text = format_percentage_points(0.25, 0.20)
    assert "pp" in diff_text
    assert "+5.0 pp" in diff_text or "+5.00 pp" in diff_text
