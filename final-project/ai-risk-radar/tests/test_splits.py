"""Deterministic unit tests for chronological partitioned and purged splits."""

import pandas as pd
import pytest

from ai_risk_radar.contracts import FeatureSet, Settings
from ai_risk_radar.splits import make_splits


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


def test_label_end_at_boundary_is_purged(sample_settings: Settings):
    """Assert a 20-session label ending at or after split boundary is purged from earlier partition."""
    # Build 10 years of daily business days ending 2026-11-16
    sessions = pd.date_range("2016-01-04", "2026-11-16", freq="B")
    n = len(sessions)

    # Build dummy feature frame
    label_ends = [sessions[i + 20] if i + 20 < n else pd.NaT for i in range(n)]
    df = pd.DataFrame(
        {
            "feature1": 1.0,
            "r20": -0.01,
            "downside20": 0,
            "label_end": label_ends,
            "is_feature_complete": True,
        },
        index=sessions,
    )
    feat_set = FeatureSet(
        frame=df,
        feature_columns=("feature1",),
        target_columns=("downside20",),
    )

    splits = make_splits(feat_set, sample_settings)

    cal_start = pd.Timestamp(splits.boundaries["calibration_start"])
    val_start = pd.Timestamp(splits.boundaries["validation_start"])
    test_start = pd.Timestamp(splits.boundaries["test_start"])

    # For any session in training, its label_end must be strictly before cal_start
    train_label_ends = df.loc[splits.train, "label_end"]
    assert (train_label_ends < cal_start).all(), "Training contains sessions whose outcomes cross into calibration"

    # For any session in calibration, its label_end must be strictly before val_start
    cal_label_ends = df.loc[splits.calibration, "label_end"]
    assert (cal_label_ends < val_start).all(), "Calibration contains sessions whose outcomes cross into validation"

    # For any session in validation, its label_end must be strictly before test_start
    val_label_ends = df.loc[splits.validation, "label_end"]
    assert (val_label_ends < test_start).all(), "Validation contains sessions whose outcomes cross into test"


def test_disjoint_partitions_and_durations(sample_settings: Settings):
    """Verify train, calibration, validation, and test partitions are disjoint and satisfy minimum history."""
    sessions = pd.date_range("2016-01-04", "2026-11-16", freq="B")
    n = len(sessions)
    label_ends = [sessions[i + 20] if i + 20 < n else pd.NaT for i in range(n)]
    df = pd.DataFrame(
        {
            "f1": 1.0,
            "r20": -0.01,
            "downside20": 0,
            "label_end": label_ends,
            "is_feature_complete": True,
        },
        index=sessions,
    )
    feat_set = FeatureSet(
        frame=df,
        feature_columns=("f1",),
        target_columns=("downside20",),
    )

    splits = make_splits(feat_set, sample_settings)

    # Disjoint check
    assert len(splits.train.intersection(splits.calibration)) == 0
    assert len(splits.train.intersection(splits.validation)) == 0
    assert len(splits.train.intersection(splits.test)) == 0
    assert len(splits.calibration.intersection(splits.validation)) == 0
    assert len(splits.calibration.intersection(splits.test)) == 0
    assert len(splits.validation.intersection(splits.test)) == 0

    # Minimum 4 years training check:
    train_duration_days = (splits.train[-1] - splits.train[0]).days
    assert train_duration_days >= 4 * 365, f"Training history duration ({train_duration_days} days) is less than 4 years"
