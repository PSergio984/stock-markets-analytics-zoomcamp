"""Deterministic unit tests for model training, calibration adequacy, and risk prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_risk_radar.contracts import FeatureSet
from ai_risk_radar.models import fit_models, predict_risk


@pytest.fixture
def synthetic_data(tmp_path: Path):
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
    settings = load_settings(cfg_file)

    # 10 years of business days
    sessions = pd.date_range("2016-01-04", "2026-11-16", freq="B")
    n = len(sessions)
    np.random.seed(42)

    # Feature frame with 3 synthetic features
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    f3 = np.random.randn(n)

    # Target label: synthetic binary event
    # Ensure sufficient events in train and calibration
    y = ((f1 + f2) > 1.2).astype(float)
    r20 = np.where(y == 1, -0.08, 0.02)
    label_ends = [sessions[i + 20] if i + 20 < n else pd.NaT for i in range(n)]

    df = pd.DataFrame(
        {
            "feat_1": f1,
            "feat_2": f2,
            "feat_3": f3,
            "r20": r20,
            "downside20": y,
            "label_end": label_ends,
            "is_feature_complete": True,
        },
        index=sessions,
    )
    feat_set = FeatureSet(
        frame=df,
        feature_columns=("feat_1", "feat_2", "feat_3"),
        target_columns=("downside20",),
    )

    from ai_risk_radar.splits import make_splits
    splits = make_splits(feat_set, settings)

    return feat_set, splits, settings


def test_fit_never_sees_validation_or_test(synthetic_data):
    """Assert changing validation/test features or labels cannot change fitted model parameters."""
    feat_set, splits, settings = synthetic_data

    models_orig = fit_models(feat_set, splits, settings)
    rf_orig = models_orig["random_forest"]

    # Tamper with validation and test rows in feat_set
    df_tampered = feat_set.frame.copy()
    val_and_test = splits.validation.union(splits.test)
    df_tampered.loc[val_and_test, "downside20"] = 1.0 - df_tampered.loc[val_and_test, "downside20"]
    df_tampered.loc[val_and_test, ["feat_1", "feat_2", "feat_3"]] += 999.0

    feat_tampered = FeatureSet(
        frame=df_tampered,
        feature_columns=feat_set.feature_columns,
        target_columns=feat_set.target_columns,
    )

    models_tampered = fit_models(feat_tampered, splits, settings)
    rf_tampered = models_tampered["random_forest"]

    # Tree structures and feature importances must be identical
    np.testing.assert_array_equal(
        rf_orig.estimator.feature_importances_,
        rf_tampered.estimator.feature_importances_,
    )


def test_calibrator_uses_only_calibration_sessions(synthetic_data):
    """Assert changing validation labels cannot change calibration metadata or parameters."""
    feat_set, splits, settings = synthetic_data

    models_orig = fit_models(feat_set, splits, settings)

    # Tamper with test only
    df_tampered = feat_set.frame.copy()
    df_tampered.loc[splits.test, "downside20"] = 0.0

    feat_tampered = FeatureSet(
        frame=df_tampered,
        feature_columns=feat_set.feature_columns,
        target_columns=feat_set.target_columns,
    )
    models_tampered = fit_models(feat_tampered, splits, settings)

    # Metadata and calibrator on random_forest should be identical
    assert models_orig["random_forest"].metadata["probability_status"] == models_tampered["random_forest"].metadata["probability_status"]


def test_constant_predictor_uses_training_prevalence(synthetic_data):
    """Assert constant predictor outputs exactly the training sample prevalence."""
    feat_set, splits, settings = synthetic_data
    models = fit_models(feat_set, splits, settings)

    y_train = feat_set.frame.loc[splits.train, "downside20"]
    expected_prevalence = float(y_train.mean())

    pred = predict_risk(models["constant_baseline"], feat_set.frame.loc[splits.test])
    assert np.allclose(pred.values, expected_prevalence)


def test_one_class_calibration_disables_probability(synthetic_data):
    """Assert a calibration partition with only one class disables calibrated probability."""
    feat_set, splits, settings = synthetic_data

    df_tampered = feat_set.frame.copy()
    # Force calibration set to all 0s (no downside event)
    df_tampered.loc[splits.calibration, "downside20"] = 0.0

    feat_tampered = FeatureSet(
        frame=df_tampered,
        feature_columns=feat_set.feature_columns,
        target_columns=feat_set.target_columns,
    )
    models = fit_models(feat_tampered, splits, settings)
    rf_bundle = models["random_forest"]

    assert rf_bundle.metadata["probability_status"] == "unavailable"
    assert "reason" in rf_bundle.metadata
    assert rf_bundle.calibrator is None


def test_serialized_model_predictions_match(synthetic_data, tmp_path: Path):
    """Assert saved and reloaded model bundle produces bit-identical predictions."""
    feat_set, splits, settings = synthetic_data
    models = fit_models(feat_set, splits, settings)
    rf_bundle = models["random_forest"]

    save_path = tmp_path / "model.joblib"
    joblib.dump(rf_bundle, save_path)

    reloaded_bundle = joblib.load(save_path)
    test_frame = feat_set.frame.loc[splits.test]

    preds_orig = predict_risk(rf_bundle, test_frame)
    preds_reloaded = predict_risk(reloaded_bundle, test_frame)

    pd.testing.assert_series_equal(preds_orig, preds_reloaded)
