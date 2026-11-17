"""Report building, JSON serialization, and atomic persistence."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ai_risk_radar.contracts import BacktestResult, Report


def _clean_json_value(val: object) -> object:
    """Recursively clean numbers to ensure strict JSON compliance (no NaN/Inf)."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, pd.Period)):
        return val.isoformat()
    if isinstance(val, dict):
        return {str(k): _clean_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_clean_json_value(v) for v in val]
    return str(val)


def build_report(
    results: dict[str, BacktestResult],
    forecast: pd.Series,
    metadata: dict[str, object],
) -> Report:
    """Construct a versioned, honest Report artifact adhering to the presentation contract."""
    # Forecast extraction
    as_of_session = (
        forecast.index[-1].strftime("%Y-%m-%d")
        if len(forecast) > 0 and isinstance(forecast.index[-1], pd.Timestamp)
        else None
    )

    prob_status = str(metadata.get("probability_status", "uncalibrated"))
    fallback_reason = metadata.get("fallback_reason")

    raw_prob = (
        float(forecast.iloc[-1])
        if len(forecast) > 0 and not pd.isna(forecast.iloc[-1])
        else None
    )

    if prob_status == "unavailable" or raw_prob is None:
        display_prob = None
        risk_level = "Unavailable"
    elif raw_prob < 0.20:
        display_prob = raw_prob
        risk_level = "Low"
    elif raw_prob < 0.35:
        display_prob = raw_prob
        risk_level = "Medium"
    else:
        display_prob = raw_prob
        risk_level = "High"

    # Allocation Determination
    tickers = list(metadata.get("tickers", ["SOXX", "XLK", "SPY", "SHY"]))
    selected_threshold = metadata.get("selected_threshold")

    weights_growth = list(metadata.get("weights_growth", [0.35, 0.35, 0.20, 0.10]))
    weights_defensive = list(metadata.get("weights_defensive", [0.10, 0.15, 0.35, 0.40]))
    weights_fixed = list(metadata.get("weights_fixed_baseline", [0.20, 0.20, 0.20, 0.40]))

    if prob_status == "unavailable" or selected_threshold is None:
        strategy_mode = "Fixed Baseline (Fallback)"
        alloc_weights = dict(zip(tickers, weights_fixed))
        trade_desc = "Probability unavailable; maintaining fixed defensive baseline allocation."
        trade_required = False
    elif raw_prob >= float(selected_threshold):
        strategy_mode = "Defensive"
        alloc_weights = dict(zip(tickers, weights_defensive))
        trade_desc = f"Risk estimate ({raw_prob:.1%}) >= threshold ({selected_threshold:.1%}); maintaining Defensive exposure."
        trade_required = True
    else:
        strategy_mode = "Growth"
        alloc_weights = dict(zip(tickers, weights_growth))
        trade_desc = f"Risk estimate ({raw_prob:.1%}) < threshold ({selected_threshold:.1%}); maintaining Growth exposure."
        trade_required = True

    # Feature explanation narrative
    explanations = metadata.get("feature_explanation", [
        "63-session peak-to-trough drawdown in AI reference basket measured.",
        "Realized 20-session volatility in SOXX and XLK incorporated into risk estimate.",
        "Relative strength difference between technology sector and broad S&P 500.",
    ])

    # Comparisons and performance dictionaries
    comparisons = {}
    for name, res in results.items():
        if name != "selected":
            comparisons[name] = res.metrics

    selected_metrics = (
        results["selected"].metrics if "selected" in results else None
    )

    limitations = [
        "Proxy limitations: SOXX and XLK represent semiconductor hardware and large-cap tech, not pure AI exposure.",
        "Execution realism: rebalance decisions on week's last session execute at next trading session close with 10 bps transaction cost.",
        "Holding drift: asset weights drift between rebalances as market prices change.",
        "Zero risk-free Sharpe convention: Sharpe ratios assume 0.0% risk-free rate strictly per project specification.",
        "Temporal overlap: 20-trading-day forecast horizons overlap, so daily observations are not statistically independent.",
    ]

    payload = {
        "identity": {
            "name": "AI Investment Risk Radar",
            "version": "0.1.0",
            "pipeline_mode": metadata.get("pipeline_mode", "research"),
            "timestamp": pd.Timestamp.now().isoformat(),
            "release_hash": metadata.get("release_hash"),
        },
        "data_status": {
            "source": metadata.get("source", "yfinance"),
            "cutoff_session": metadata.get("cutoff_session", as_of_session),
            "as_of_session": as_of_session,
            "staleness_sessions": metadata.get("staleness_sessions", 0),
            "refresh_status": metadata.get("refresh_status", "ok"),
        },
        "forecast": {
            "as_of_session": as_of_session,
            "horizon_sessions": 20,
            "loss_threshold": -0.05,
            "loss_definition": "Estimated probability of 50% SOXX / 50% XLK reference basket total return <= -5.0% over next 20 trading sessions",
            "probability": display_prob,
            "probability_status": prob_status,
            "fallback_reason": fallback_reason,
            "raw_probability": raw_prob,
            "risk_level": risk_level,
            "feature_explanation": explanations,
        },
        "allocation": {
            "strategy_mode": strategy_mode,
            "selected_threshold": selected_threshold,
            "weights": alloc_weights,
            "trade_required": trade_required,
            "trade_description": trade_desc,
        },
        "performance": {
            "evaluation_partition": metadata.get("evaluation_partition", "validation"),
            "selected_strategy": selected_metrics,
        },
        "comparisons": comparisons,
        "diagnostics": metadata.get("diagnostics", {}),
        "limitations": limitations,
    }

    clean_payload = _clean_json_value(payload)
    return Report(payload=clean_payload)


def save_report(report: Report, path: Path | str) -> Path:
    """Atomically save report JSON to disk using a temporary file and rename."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp")
    text_content = json.dumps(report.payload, indent=2)

    tmp_path.write_text(text_content, encoding="utf-8")
    tmp_path.replace(out_path)

    return out_path


def load_report(path: Path | str) -> Report:
    """Load a Report JSON artifact from disk."""
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")

    data = json.loads(report_path.read_text(encoding="utf-8"))
    return Report(payload=data)
