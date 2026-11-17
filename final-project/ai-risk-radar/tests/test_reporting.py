"""Deterministic unit tests for report generation, serialization, and atomic persistence."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.contracts import BacktestResult
from ai_risk_radar.reporting import build_report, load_report, save_report


def test_missing_report_is_explicit(tmp_path: Path):
    """Assert attempting to load a non-existent report raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent_report.json"
    with pytest.raises(FileNotFoundError, match="Report not found"):
        load_report(missing_path)


def test_unavailable_probability_uses_fixed_baseline():
    """Assert unavailable probability status forces allocation to fixed baseline fallback."""
    b_result = BacktestResult(
        daily=pd.DataFrame({"equity": [10000.0, 10100.0]}, index=pd.date_range("2024-01-02", periods=2, freq="B")),
        trades=pd.DataFrame(),
        metrics={"cagr": 0.10, "max_drawdown": -0.05, "sharpe": 1.2},
    )
    forecast_series = pd.Series([0.45], index=[pd.Timestamp("2024-01-03")])
    metadata = {
        "pipeline_mode": "latest",
        "probability_status": "unavailable",
        "fallback_reason": "Inadequate positive events in calibration split",
        "selected_threshold": None,
        "weights_fixed_baseline": [0.20, 0.20, 0.20, 0.40],
        "weights_growth": [0.35, 0.35, 0.20, 0.10],
        "weights_defensive": [0.10, 0.15, 0.35, 0.40],
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
    }

    report = build_report(
        results={"selected": b_result},
        forecast=forecast_series,
        metadata=metadata,
    )

    payload = report.payload
    assert payload["forecast"]["probability_status"] == "unavailable"
    assert payload["allocation"]["strategy_mode"] == "Fixed Baseline (Fallback)"
    assert payload["allocation"]["weights"]["SHY"] == 0.40


def test_serialized_report_has_no_nonfinite_json(tmp_path: Path):
    """Assert serialized JSON contains no NaN or Infinity values."""
    b_result = BacktestResult(
        daily=pd.DataFrame({"equity": [10000.0]}, index=[pd.Timestamp("2024-01-02")]),
        trades=pd.DataFrame(),
        metrics={"cagr": None, "max_drawdown": 0.0, "sharpe": None},  # Nulls
    )
    forecast_series = pd.Series([np.nan], index=[pd.Timestamp("2024-01-02")])
    metadata = {
        "pipeline_mode": "research",
        "probability_status": "calibrated",
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
        "weights_growth": [0.35, 0.35, 0.20, 0.10],
        "weights_defensive": [0.10, 0.15, 0.35, 0.40],
        "weights_fixed_baseline": [0.20, 0.20, 0.20, 0.40],
    }

    report = build_report({"selected": b_result}, forecast_series, metadata)
    save_path = tmp_path / "report.json"
    save_report(report, save_path)

    raw_text = save_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    loaded = load_report(save_path)
    assert loaded.payload["performance"]["selected_strategy"]["cagr"] is None
