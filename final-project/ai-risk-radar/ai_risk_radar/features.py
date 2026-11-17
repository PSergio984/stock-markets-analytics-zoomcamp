"""Causal feature engineering and exact downside target construction."""

import numpy as np
import pandas as pd

from ai_risk_radar.contracts import DataBundle, FeatureSet, Settings


def build_features(bundle: DataBundle, settings: Settings) -> FeatureSet:
    """Build causal price-derived technical features and exact 20-session downside labels."""
    prices = bundle.adjusted_close.copy()
    sessions = bundle.sessions
    n = len(sessions)

    # 1. Target Construction:
    # 50% SOXX / 50% XLK held through each 20-session horizon
    # r20(t) = 0.5 * (P_SOXX(t+20)/P_SOXX(t) - 1) + 0.5 * (P_XLK(t+20)/P_XLK(t) - 1)
    horizon = settings.target_horizon
    soxx = prices["SOXX"]
    xlk = prices["XLK"]

    soxx_fwd_ret = (soxx.shift(-horizon) / soxx) - 1.0
    xlk_fwd_ret = (xlk.shift(-horizon) / xlk) - 1.0
    r20 = 0.5 * soxx_fwd_ret + 0.5 * xlk_fwd_ret

    downside20 = pd.Series(np.nan, index=sessions, dtype=float)
    valid_mask = ~r20.isna()
    downside20[valid_mask] = (r20[valid_mask] <= settings.target_threshold).astype(float)

    # Label end timestamp: session at t + horizon
    label_ends = [
        sessions[i + horizon] if i + horizon < n else pd.NaT
        for i in range(n)
    ]
    label_end_series = pd.Series(label_ends, index=sessions, dtype="datetime64[ns]")

    # 2. Causal Feature Engineering (Strictly backward-looking):
    features = pd.DataFrame(index=sessions)

    # Normalized 50/50 basket index for composite features
    basket_series = 0.5 * (soxx / soxx.iloc[0]) + 0.5 * (xlk / xlk.iloc[0])
    all_series = {
        "soxx": soxx,
        "xlk": xlk,
        "spy": prices["SPY"],
        "shy": prices["SHY"],
        "basket": basket_series,
    }

    # Daily returns (1-day)
    daily_rets = {k: s.pct_change(1) for k, s in all_series.items()}

    # Lookbacks: 1, 5, 20, 63 sessions returns
    for name, s in all_series.items():
        features[f"ret_{name}_1"] = s.pct_change(1)
        features[f"ret_{name}_5"] = s.pct_change(5)
        features[f"ret_{name}_20"] = s.pct_change(20)
        features[f"ret_{name}_63"] = s.pct_change(63)

    # Realized Volatilities: 20 and 63 sessions (annualized with sqrt(252))
    for name, d_ret in daily_rets.items():
        features[f"vol_{name}_20"] = d_ret.rolling(20).std() * np.sqrt(252)
        features[f"vol_{name}_63"] = d_ret.rolling(63).std() * np.sqrt(252)

    # Drawdown from 63-session peak: P / max(P, 63) - 1
    for name, s in all_series.items():
        if name in ("soxx", "xlk", "spy", "basket"):
            rolling_max = s.rolling(63).max()
            features[f"dd_{name}_63"] = (s / rolling_max) - 1.0

    # Relative Strength against SPY: difference in returns
    for name in ("soxx", "xlk", "basket"):
        features[f"rel_{name}_spy_20"] = features[f"ret_{name}_20"] - features["ret_spy_20"]
        features[f"rel_{name}_spy_63"] = features[f"ret_{name}_63"] - features["ret_spy_63"]

    # Rolling correlations (63-session)
    features["corr_soxx_spy_63"] = daily_rets["soxx"].rolling(63).corr(daily_rets["spy"])
    features["corr_xlk_spy_63"] = daily_rets["xlk"].rolling(63).corr(daily_rets["spy"])
    features["corr_soxx_xlk_63"] = daily_rets["soxx"].rolling(63).corr(daily_rets["xlk"])

    feature_columns = tuple(features.columns)

    # Check completeness of all features (not NaN)
    is_feature_complete = ~features.isna().any(axis=1)

    # Combine into full frame
    full_frame = features.copy()
    full_frame["r20"] = r20
    full_frame["downside20"] = downside20
    full_frame["label_end"] = label_end_series
    full_frame["is_feature_complete"] = is_feature_complete

    target_columns = ("downside20",)

    return FeatureSet(
        frame=full_frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
