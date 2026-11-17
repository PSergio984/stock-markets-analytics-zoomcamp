"""Chronological partitioned and purged dataset splits."""

import pandas as pd

from ai_risk_radar.contracts import FeatureSet, Settings, SplitSet


def make_splits(features: FeatureSet, settings: Settings) -> SplitSet:
    """Create chronological purged partitions: train (>=4y), cal (12m), val (24m), test (24m)."""
    df = features.frame
    sessions = pd.DatetimeIndex(df.index)
    cutoff = sessions[-1]

    # Calculate partition boundaries anchored to the cutoff session
    test_start_target = cutoff - pd.DateOffset(months=24)
    test_sessions = sessions[sessions >= test_start_target]
    if len(test_sessions) == 0:
        raise ValueError("Insufficient sessions for 24-month test split")
    test_start = test_sessions[0]

    val_start_target = test_start - pd.DateOffset(months=24)
    val_sessions = sessions[(sessions >= val_start_target) & (sessions < test_start)]
    if len(val_sessions) == 0:
        raise ValueError("Insufficient sessions for 24-month validation split")
    val_start = val_sessions[0]

    cal_start_target = val_start - pd.DateOffset(months=12)
    cal_sessions = sessions[(sessions >= cal_start_target) & (sessions < val_start)]
    if len(cal_sessions) == 0:
        raise ValueError("Insufficient sessions for 12-month calibration split")
    cal_start = cal_sessions[0]

    # Training is everything prior to cal_start with complete features
    train_start = sessions[0]
    train_duration_days = (cal_start - train_start).days
    if train_duration_days < 4 * 365:
        raise RuntimeError(
            f"Training history must be at least 4 years (1460 days), got {train_duration_days} days"
        )

    # Apply Purging:
    # A session t in a partition is purged if label_end(t) >= next_partition_start
    # or if features are incomplete or label is NaN.

    # 1. Train purged:
    train_mask = (
        (df.index >= train_start)
        & (df.index < cal_start)
        & df["is_feature_complete"]
        & ~df["downside20"].isna()
        & (df["label_end"] < cal_start)
    )
    train_idx = pd.DatetimeIndex(df.index[train_mask])

    # 2. Calibration purged:
    cal_mask = (
        (df.index >= cal_start)
        & (df.index < val_start)
        & df["is_feature_complete"]
        & ~df["downside20"].isna()
        & (df["label_end"] < val_start)
    )
    cal_idx = pd.DatetimeIndex(df.index[cal_mask])

    # 3. Validation purged:
    val_mask = (
        (df.index >= val_start)
        & (df.index < test_start)
        & df["is_feature_complete"]
        & ~df["downside20"].isna()
        & (df["label_end"] < test_start)
    )
    val_idx = pd.DatetimeIndex(df.index[val_mask])

    # 4. Test: all sessions in test period with complete features (both labeled and latest unlabeled)
    test_mask = (df.index >= test_start) & (df.index <= cutoff) & df["is_feature_complete"]
    test_idx = pd.DatetimeIndex(df.index[test_mask])

    boundaries = {
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": cal_start.strftime("%Y-%m-%d"),
        "calibration_start": cal_start.strftime("%Y-%m-%d"),
        "calibration_end": val_start.strftime("%Y-%m-%d"),
        "validation_start": val_start.strftime("%Y-%m-%d"),
        "validation_end": test_start.strftime("%Y-%m-%d"),
        "test_start": test_start.strftime("%Y-%m-%d"),
        "test_end": cutoff.strftime("%Y-%m-%d"),
        "cutoff": cutoff.strftime("%Y-%m-%d"),
    }

    return SplitSet(
        train=train_idx,
        calibration=cal_idx,
        validation=val_idx,
        test=test_idx,
        boundaries=boundaries,
    )
