"""Financial performance metrics: CAGR, Peak-to-Trough Drawdown, Sharpe (rf=0), and Turnover."""

import numpy as np
import pandas as pd


def compute_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float | None = None,
) -> dict[str, float | None]:
    """Compute annualized CAGR, max drawdown, Sharpe (rf=0), volatility, turnover, and occupancy."""
    if daily.empty:
        return {
            "total_return": None,
            "cagr": None,
            "max_drawdown": None,
            "annualized_volatility": None,
            "sharpe": None,
            "turnover": None,
            "defensive_occupancy": None,
            "total_cost": None,
        }

    equity = daily["equity"].values
    n = len(equity)
    if n < 2 or equity[0] <= 0:
        return {
            "total_return": None,
            "cagr": None,
            "max_drawdown": None,
            "annualized_volatility": None,
            "sharpe": None,
            "turnover": None,
            "defensive_occupancy": None,
            "total_cost": None,
        }

    v_start = float(initial_capital) if initial_capital is not None and initial_capital > 0 else float(equity[0])
    v_end = float(equity[-1])
    total_ret = float((v_end - v_start) / v_start)

    # CAGR: (V_end / V_start) ** (252 / (N - 1)) - 1
    years = (n - 1) / 252.0
    cagr = float((v_end / v_start) ** (1.0 / years) - 1.0) if years > 0 and v_start > 0 and v_end > 0 else None

    # Max Drawdown
    if initial_capital is not None and initial_capital > 0:
        # Include initial capital as the peak baseline prior to initial funding cost
        full_series = np.concatenate([[float(initial_capital)], equity])
        peaks = np.maximum.accumulate(full_series)
        drawdowns = (full_series - peaks) / peaks
        max_dd = float(np.min(drawdowns))
    else:
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        max_dd = float(np.min(drawdowns))

    # Daily net returns
    if "net_return" in daily.columns:
        net_rets = daily["net_return"].dropna().values
    else:
        net_rets = np.diff(equity) / equity[:-1]

    if len(net_rets) > 1 and np.std(net_rets) > 1e-9:
        ann_vol = float(np.std(net_rets) * np.sqrt(252.0))
        mean_ret = float(np.mean(net_rets))
        # Zero risk-free rate assumption strictly per specification
        sharpe = float(np.sqrt(252.0) * mean_ret / np.std(net_rets))
    else:
        ann_vol = 0.0
        sharpe = None

    # Annualized turnover
    if not trades.empty and "turnover" in trades.columns:
        total_turnover = float(trades["turnover"].sum())
        ann_turnover = float(total_turnover / max(years, 0.01))
        total_cost = float(trades["cost"].sum()) if "cost" in trades.columns else 0.0
    else:
        ann_turnover = 0.0
        total_cost = 0.0

    # Defensive occupancy: fraction of days where SHY weight > 0.25 (growth has 10% SHY, defensive has 40% SHY)
    if "holding_SHY" in daily.columns:
        shy_weights = daily["holding_SHY"].values / np.maximum(equity, 1e-9)
        def_days = np.sum(shy_weights > 0.25)
        def_occupancy = float(def_days / n)
    else:
        def_occupancy = 0.0

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "turnover": ann_turnover,
        "defensive_occupancy": def_occupancy,
        "total_cost": total_cost,
    }
