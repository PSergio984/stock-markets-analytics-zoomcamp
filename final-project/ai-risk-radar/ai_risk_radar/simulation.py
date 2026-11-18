"""Execution simulation with trade delay, drifted holdings, and exact transaction costs."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ai_risk_radar.contracts import BacktestResult, DataBundle, Settings
from ai_risk_radar.metrics import compute_metrics


def _solve_post_trade_equity(
    v_pre: float,
    h_pre: np.ndarray,
    target_weights: np.ndarray,
    cost_rate: float,
) -> float:
    """Solve V_post = V_pre - c * sum(|w * V_post - h_pre|) to conserve equity under costs."""
    if cost_rate <= 0.0:
        return v_pre

    if np.all(h_pre == 0.0):
        # Initial cash funding: V_post = V_pre / (1 + c)
        return v_pre / (1.0 + cost_rate)

    def obj(v: float) -> float:
        cost = cost_rate * np.sum(np.abs(target_weights * v - h_pre))
        return v + cost - v_pre

    # Bracket search between 0 and v_pre
    try:
        v_post = brentq(obj, 0.0, v_pre, xtol=1e-12)
    except ValueError:
        # Fallback if v_pre is extremely close to 0
        v_post = v_pre / (1.0 + cost_rate)

    return float(v_post)


def weekly_targets(
    risk: pd.Series,
    bundle_or_sessions: DataBundle | pd.DatetimeIndex | pd.DataFrame,
    settings: Settings,
    threshold: float,
) -> pd.DataFrame:
    """Generate weekly target allocations with signal on week's last session and execution next session close."""
    if isinstance(bundle_or_sessions, DataBundle):
        sessions = pd.DatetimeIndex(bundle_or_sessions.sessions)
    elif isinstance(bundle_or_sessions, pd.DataFrame):
        sessions = pd.DatetimeIndex(bundle_or_sessions.index)
    else:
        sessions = pd.DatetimeIndex(bundle_or_sessions)

    tickers = list(settings.tickers)
    targets = pd.DataFrame(np.nan, index=sessions, columns=tickers)

    # Initial funding on session 0 of this simulated period
    targets.iloc[0] = list(settings.weights_growth)

    # Group sessions by calendar year and ISO week
    iso_cal = pd.DataFrame(
        {
            "year": sessions.isocalendar().year,
            "week": sessions.isocalendar().week,
        },
        index=sessions,
    )

    grouped = iso_cal.groupby(["year", "week"])
    for _, group in grouped:
        signal_session = group.index[-1]  # Last completed trading session of the week
        if signal_session in risk.index and not pd.isna(risk.loc[signal_session]):
            risk_val = float(risk.loc[signal_session])
            weights = (
                settings.weights_defensive
                if risk_val >= threshold
                else settings.weights_growth
            )

            # Find next trading session in sessions for execution
            future_sessions = sessions[sessions > signal_session]
            if len(future_sessions) > 0:
                exec_session = future_sessions[0]
                targets.loc[exec_session] = list(weights)

    return targets


def simulate(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    settings: Settings,
    cost_rate: float,
) -> BacktestResult:
    """Simulate delayed allocation portfolio with daily drifted weights and exact transaction costs."""
    sessions = prices.index
    n = len(sessions)
    tickers = list(settings.tickers)
    k = len(tickers)

    daily_records = []
    trade_records = []

    h_post = np.zeros(k, dtype=float)
    v_post = float(settings.initial_capital)

    for i in range(n):
        dt = sessions[i]
        curr_prices = prices.loc[dt, tickers].values

        if i > 0:
            prev_prices = prices.iloc[i - 1][tickers].values
            asset_returns = (curr_prices / prev_prices) - 1.0
            h_pre = h_post * (1.0 + asset_returns)
            v_pre = float(np.sum(h_pre))
            gross_return = (v_pre - v_post) / v_post if v_post > 0 else 0.0
        else:
            h_pre = np.zeros(k, dtype=float)
            v_pre = float(settings.initial_capital)
            gross_return = 0.0

        # Check for rebalance target on this session
        target_row = targets.loc[dt, tickers] if dt in targets.index else pd.Series(np.nan, index=tickers)
        if i == 0 and target_row.isna().all():
            target_weights = np.array(settings.weights_growth, dtype=float)
            should_rebalance = True
        elif not target_row.isna().all():
            target_weights = target_row.values.astype(float)
            should_rebalance = True
        else:
            target_weights = h_pre / v_pre if v_pre > 0 else np.array(settings.weights_growth, dtype=float)
            should_rebalance = False

        if should_rebalance:
            v_post_new = _solve_post_trade_equity(v_pre, h_pre, target_weights, cost_rate)
            h_post_new = target_weights * v_post_new
            cost_charged = v_pre - v_post_new
            dollars_traded = float(np.sum(np.abs(h_post_new - h_pre)))
            turnover = dollars_traded / v_pre if v_pre > 0 else 0.0

            trade_records.append({
                "exec_date": dt,
                "v_pre": v_pre,
                "v_post": v_post_new,
                "cost": cost_charged,
                "turnover": turnover,
            })
            h_post = h_post_new
            v_post_prev = v_post
            v_post = v_post_new
        else:
            h_post = h_pre
            v_post_prev = v_post
            v_post = v_pre
            cost_charged = 0.0

        net_return = (v_post - v_post_prev) / v_post_prev if i > 0 and v_post_prev > 0 else (v_post - settings.initial_capital) / settings.initial_capital

        rec = {
            "equity": v_post,
            "gross_return": gross_return,
            "net_return": net_return,
            "cost": cost_charged,
        }
        for j, ticker in enumerate(tickers):
            rec[f"holding_{ticker}"] = h_post[j]

        daily_records.append(rec)

    daily_df = pd.DataFrame(daily_records, index=sessions)
    trades_df = pd.DataFrame(trade_records)

    metrics = compute_metrics(daily_df, trades_df, initial_capital=settings.initial_capital)

    return BacktestResult(
        daily=daily_df,
        trades=trades_df,
        metrics=metrics,
    )


def benchmark_targets(
    bundle: DataBundle,
    settings: Settings,
    sessions: pd.DatetimeIndex | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate benchmark allocation schedules: growth buy-and-hold, fixed defensive, and momentum."""
    target_sessions = pd.DatetimeIndex(bundle.sessions) if sessions is None else pd.DatetimeIndex(sessions)
    tickers = list(settings.tickers)

    # 1. Growth Buy and Hold: trade once at session 0 of this period, then hold and drift
    bnh = pd.DataFrame(np.nan, index=target_sessions, columns=tickers)
    bnh.iloc[0] = list(settings.weights_growth)

    # 2. Fixed Defensive Weekly: rebalance weekly to fixed baseline
    defensive_weekly = pd.DataFrame(np.nan, index=target_sessions, columns=tickers)
    defensive_weekly.iloc[0] = list(settings.weights_fixed_baseline)

    iso_cal = pd.DataFrame(
        {"year": target_sessions.isocalendar().year, "week": target_sessions.isocalendar().week},
        index=target_sessions,
    )
    for _, group in iso_cal.groupby(["year", "week"]):
        sig_dt = group.index[-1]
        future_sess = target_sessions[target_sessions > sig_dt]
        if len(future_sess) > 0:
            defensive_weekly.loc[future_sess[0]] = list(settings.weights_fixed_baseline)

    # 3. Trailing 63-Session Basket Momentum Policy
    mom = pd.DataFrame(np.nan, index=target_sessions, columns=tickers)
    mom.iloc[0] = list(settings.weights_growth)

    prices = bundle.adjusted_close
    basket = 0.5 * (prices["SOXX"] / prices["SOXX"].iloc[0]) + 0.5 * (prices["XLK"] / prices["XLK"].iloc[0])
    basket_ret63 = basket.pct_change(63)

    for _, group in iso_cal.groupby(["year", "week"]):
        sig_dt = group.index[-1]
        if sig_dt in basket_ret63.index and not pd.isna(basket_ret63.loc[sig_dt]):
            # If 63-session momentum is positive, choose Growth; else Defensive
            w = (
                settings.weights_growth
                if basket_ret63.loc[sig_dt] >= 0.0
                else settings.weights_defensive
            )
            future_sess = target_sessions[target_sessions > sig_dt]
            if len(future_sess) > 0:
                mom.loc[future_sess[0]] = list(w)

    return {
        "growth_buy_and_hold": bnh,
        "fixed_defensive_weekly": defensive_weekly,
        "momentum_63": mom,
    }


def select_strategy(
    candidates: dict[str, BacktestResult],
    baseline: BacktestResult,
) -> str | None:
    """Select the candidate with smallest max-drawdown magnitude whose CAGR >= baseline's CAGR."""
    baseline_cagr = baseline.metrics.get("cagr")
    if baseline_cagr is None:
        return None

    qualifying = []
    for name, res in candidates.items():
        cagr = res.metrics.get("cagr")
        max_dd = res.metrics.get("max_drawdown")
        turnover = res.metrics.get("turnover", 0.0) or 0.0

        if cagr is not None and max_dd is not None and cagr >= baseline_cagr:
            qualifying.append((abs(max_dd), -cagr, turnover, name))

    if not qualifying:
        return None

    # Sort by: smallest abs(max_drawdown), highest CAGR (-cagr), lowest turnover
    qualifying.sort()
    return qualifying[0][3]
