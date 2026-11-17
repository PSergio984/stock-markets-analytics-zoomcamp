"""Downside risk model fitting, cross-validation, calibration, and evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ai_risk_radar.contracts import FeatureSet, ModelBundle, Settings, SplitSet


class ConstantPredictor:
    """Baseline model predicting constant historical training event prevalence."""

    def __init__(self, prevalence: float):
        self.prevalence = float(prevalence)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n = len(X)
        p = np.full((n, 2), [1.0 - self.prevalence, self.prevalence])
        return p


class SigmoidCalibrator:
    """Platt scaling sigmoid calibrator mapping raw model probabilities to calibrated risk probabilities."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1.0, solver="lbfgs")

    def fit(self, raw_probas: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        p = np.clip(raw_probas, 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.lr.fit(logit, y)
        return self

    def predict_proba(self, raw_probas: np.ndarray) -> np.ndarray:
        p = np.clip(raw_probas, 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.lr.predict_proba(logit)[:, 1]


def _check_calibration_support(
    y_cal: pd.Series,
    sessions_cal: pd.DatetimeIndex,
) -> tuple[bool, str]:
    """Check minimum calibration support: >=20 per class, >=2 positive starts separated by 20 sessions."""
    counts = y_cal.value_counts()
    count_0 = counts.get(0.0, 0)
    count_1 = counts.get(1.0, 0)

    if count_0 < 20 or count_1 < 20:
        return False, f"Inadequate class support in calibration split: class 0={count_0}, class 1={count_1} (need >= 20 each)"

    # Check for at least two positive events separated by at least 20 sessions
    pos_mask = y_cal == 1.0
    pos_indices = np.where(pos_mask.values)[0]
    if len(pos_indices) < 2:
        return False, "Fewer than 2 positive event sessions in calibration split"

    max_gap = 0
    for i in range(len(pos_indices)):
        for j in range(i + 1, len(pos_indices)):
            gap = pos_indices[j] - pos_indices[i]
            max_gap = max(max_gap, gap)
            if gap >= 20:
                return True, "Calibration support adequate"

    return False, f"Positive events not separated by >=20 sessions (maximum session gap={max_gap})"


def fit_models(
    features: FeatureSet,
    splits: SplitSet,
    settings: Settings,
) -> dict[str, ModelBundle]:
    """Train constant baseline, Decision Tree, and Random Forest models on training split."""
    df = features.frame
    feature_cols = list(features.feature_columns)
    target_col = features.target_columns[0]

    # Strictly training split data
    X_train = df.loc[splits.train, feature_cols]
    y_train = df.loc[splits.train, target_col]

    # Calibration split data
    X_cal = df.loc[splits.calibration, feature_cols]
    y_cal = df.loc[splits.calibration, target_col]

    models: dict[str, ModelBundle] = {}

    # 1. Constant Prevalence Baseline
    train_prev = float(y_train.mean())
    const_est = ConstantPredictor(train_prev)
    models["constant_baseline"] = ModelBundle(
        estimator=const_est,
        calibrator=None,
        feature_columns=features.feature_columns,
        metadata={
            "model_name": "constant_baseline",
            "training_prevalence": train_prev,
            "probability_status": "uncalibrated",
            "fit_sessions": len(splits.train),
        },
    )

    # 2. Decision Tree Model Selection (3 expanding purged folds)
    dt_candidates = [2, 3]
    best_dt_depth = dt_candidates[0]
    best_dt_brier = float("inf")

    n_train = len(splits.train)
    q1 = n_train // 4
    q2 = n_train // 2
    q3 = (3 * n_train) // 4

    folds = [
        (0, max(1, q1 - 20), q1, q2),
        (0, max(1, q2 - 20), q2, q3),
        (0, max(1, q3 - 20), q3, n_train),
    ]

    for depth in dt_candidates:
        fold_briers = []
        for tr_s, tr_e, val_s, val_e in folds:
            fold_tr_idx = splits.train[tr_s:tr_e]
            fold_val_idx = splits.train[val_s:val_e]
            clf = DecisionTreeClassifier(max_depth=depth, random_state=settings.random_seed)
            clf.fit(df.loc[fold_tr_idx, feature_cols], df.loc[fold_tr_idx, target_col])
            p_val = clf.predict_proba(df.loc[fold_val_idx, feature_cols])[:, 1]
            fold_briers.append(np.mean((df.loc[fold_val_idx, target_col] - p_val) ** 2))
        avg_brier = float(np.mean(fold_briers))
        if avg_brier < best_dt_brier:
            best_dt_brier = avg_brier
            best_dt_depth = depth

    final_dt = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=settings.random_seed)
    final_dt.fit(X_train, y_train)

    models["decision_tree"] = ModelBundle(
        estimator=final_dt,
        calibrator=None,
        feature_columns=features.feature_columns,
        metadata={
            "model_name": "decision_tree",
            "selected_depth": best_dt_depth,
            "cv_brier_score": best_dt_brier,
            "probability_status": "uncalibrated",
            "fit_sessions": len(splits.train),
        },
    )

    # 3. Random Forest Model Selection (3 expanding purged folds)
    rf_candidates = [
        (depth, min_leaf)
        for depth in [3, 5]
        for min_leaf in [20, 50]
    ]
    best_rf_params = rf_candidates[0]
    best_rf_brier = float("inf")

    for depth, min_leaf in rf_candidates:
        fold_briers = []
        for tr_s, tr_e, val_s, val_e in folds:
            fold_tr_idx = splits.train[tr_s:tr_e]
            fold_val_idx = splits.train[val_s:val_e]
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=settings.random_seed,
                n_jobs=-1,
            )
            clf.fit(df.loc[fold_tr_idx, feature_cols], df.loc[fold_tr_idx, target_col])
            p_val = clf.predict_proba(df.loc[fold_val_idx, feature_cols])[:, 1]
            fold_briers.append(np.mean((df.loc[fold_val_idx, target_col] - p_val) ** 2))
        avg_brier = float(np.mean(fold_briers))
        if avg_brier < best_rf_brier:
            best_rf_brier = avg_brier
            best_rf_params = (depth, min_leaf)

    final_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=best_rf_params[0],
        min_samples_leaf=best_rf_params[1],
        random_state=settings.random_seed,
        n_jobs=-1,
    )
    final_rf.fit(X_train, y_train)

    # Sigmoid calibration check and fit on calibration split
    cal_ok, cal_reason = _check_calibration_support(y_cal, splits.calibration)
    calibrator = None
    prob_status = "unavailable"

    if cal_ok:
        raw_cal_p = final_rf.predict_proba(X_cal)[:, 1]
        calibrator = SigmoidCalibrator()
        calibrator.fit(raw_cal_p, y_cal.values)
        prob_status = "calibrated"

    models["random_forest"] = ModelBundle(
        estimator=final_rf,
        calibrator=calibrator,
        feature_columns=features.feature_columns,
        metadata={
            "model_name": "random_forest",
            "selected_max_depth": best_rf_params[0],
            "selected_min_samples_leaf": best_rf_params[1],
            "cv_brier_score": best_rf_brier,
            "probability_status": prob_status,
            "reason": cal_reason,
            "fit_sessions": len(splits.train),
            "calibration_sessions": len(splits.calibration),
        },
    )

    return models


def predict_risk(bundle: ModelBundle, frame: pd.DataFrame) -> pd.Series:
    """Generate risk estimate series for the given feature dataframe."""
    feature_cols = list(bundle.feature_columns)
    X = frame[feature_cols]

    if isinstance(bundle.estimator, ConstantPredictor):
        p = np.full(len(frame), bundle.estimator.prevalence)
        return pd.Series(p, index=frame.index, name="risk_probability")

    raw_proba = bundle.estimator.predict_proba(X)[:, 1]
    if bundle.calibrator is not None and bundle.metadata.get("probability_status") == "calibrated":
        proba = bundle.calibrator.predict_proba(raw_proba)
    else:
        proba = raw_proba

    return pd.Series(proba, index=frame.index, name="risk_probability")


def classification_report(y: pd.Series, probability: pd.Series) -> dict[str, object]:
    """Compute classification metrics, Brier score, and reliability bins with non-overlapping subsampling."""
    mask = ~y.isna() & ~probability.isna()
    y_clean = y[mask].astype(float)
    p_clean = probability[mask].astype(float)

    n = len(y_clean)
    if n == 0:
        return {"brier_score": None, "prevalence": None, "count": 0}

    brier = float(np.mean((y_clean - p_clean) ** 2))
    prev = float(np.mean(y_clean))

    # Threshold precision & recall
    threshold_metrics = {}
    for th in [0.20, 0.35, 0.50]:
        th_key = f"{int(th * 100)}"
        pred_pos = p_clean >= th
        act_pos = y_clean == 1.0

        tp = int(np.sum(pred_pos & act_pos))
        fp = int(np.sum(pred_pos & ~act_pos))
        fn = int(np.sum(~pred_pos & act_pos))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None

        threshold_metrics[f"precision_{th_key}"] = precision
        threshold_metrics[f"recall_{th_key}"] = recall

    # Reliability Bins (5 equal width bins from 0 to 1)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    reliability_bins = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        in_bin = (p_clean >= low) & (p_clean <= high if i == len(bins) - 2 else p_clean < high)
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            pred_mean = float(np.mean(p_clean[in_bin]))
            obs_mean = float(np.mean(y_clean[in_bin]))
        else:
            pred_mean = None
            obs_mean = None
        reliability_bins.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "count": bin_count,
            "pred_mean": pred_mean,
            "obs_mean": obs_mean,
        })

    # Non-overlapping evaluation (every 20th session)
    sub_y = y_clean.iloc[::20]
    sub_p = p_clean.iloc[::20]
    sub_brier = float(np.mean((sub_y - sub_p) ** 2)) if len(sub_y) > 0 else None
    sub_prev = float(np.mean(sub_y)) if len(sub_y) > 0 else None
    sub_pos = int(np.sum(sub_y == 1.0))

    report = {
        "brier_score": brier,
        "prevalence": prev,
        "count": n,
        **threshold_metrics,
        "reliability_bins": reliability_bins,
        "non_overlapping": {
            "brier_score": sub_brier,
            "prevalence": sub_prev,
            "event_count": sub_pos,
            "total_count": len(sub_y),
        },
    }

    return report
