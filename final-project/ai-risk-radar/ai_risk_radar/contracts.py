"""Typed data contracts and records for AI Investment Risk Radar."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Settings:
    """Validated runtime and evaluation settings."""

    tickers: tuple[str, ...]
    acquisition_start: str
    source: str
    target_horizon: int
    target_threshold: float
    weights_growth: tuple[float, ...]
    weights_defensive: tuple[float, ...]
    weights_fixed_baseline: tuple[float, ...]
    candidate_thresholds: tuple[float, ...]
    cost_rate: float
    cost_sensitivities: tuple[float, ...]
    random_seed: int
    initial_capital: float
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class DataBundle:
    """Standardized price snapshot and ingestion manifest."""

    adjusted_close: pd.DataFrame
    volume: pd.DataFrame
    sessions: pd.DatetimeIndex
    manifest: dict[str, object]


@dataclass(frozen=True)
class FeatureSet:
    """Causal features, target horizons, and completeness flags."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class SplitSet:
    """Chronologically partitioned and purged trading session indices."""

    train: pd.DatetimeIndex
    calibration: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex
    boundaries: dict[str, str]


@dataclass(frozen=True)
class ModelBundle:
    """Trained estimator, probability calibrator, and fit metadata."""

    estimator: object
    calibrator: object | None
    feature_columns: tuple[str, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class BacktestResult:
    """Simulation output with daily holdings, trades, and portfolio metrics."""

    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | None]


@dataclass(frozen=True)
class Report:
    """Presentation artifact serializable to JSON without NaN or Inf values."""

    payload: dict[str, object]
