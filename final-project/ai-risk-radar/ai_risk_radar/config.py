"""Configuration loading and validation for AI Investment Risk Radar."""

import json
import math
from pathlib import Path

from ai_risk_radar.contracts import Settings


def load_settings(path: Path | str) -> Settings:
    """Load and validate settings from a JSON file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tickers = tuple(data["tickers"])
    expected_tickers = ("SOXX", "XLK", "SPY", "SHY")
    if tickers != expected_tickers:
        raise ValueError(f"Tickers must be {expected_tickers}, got {tickers}")

    weights = data.get("weights", {})
    growth = tuple(float(w) for w in weights["growth"])
    defensive = tuple(float(w) for w in weights["defensive"])
    fixed_baseline = tuple(float(w) for w in weights["fixed_baseline"])

    for name, w_vec in [
        ("growth", growth),
        ("defensive", defensive),
        ("fixed_baseline", fixed_baseline),
    ]:
        if len(w_vec) != 4:
            raise ValueError(f"Weight vector '{name}' must have 4 elements, got {len(w_vec)}")
        if any(math.isnan(w) or w < 0.0 for w in w_vec):
            raise ValueError(f"Weight vector '{name}' contains NaN or negative weights")
        if math.isnan(sum(w_vec)) or abs(sum(w_vec) - 1.0) > 1e-5:
            raise ValueError(f"Weights for '{name}' must sum to 1.0, got {sum(w_vec)}")

    candidate_thresholds = tuple(float(t) for t in data["candidate_thresholds"])
    cost_sensitivities = tuple(float(c) for c in data.get("cost_sensitivities", [0.0005, 0.0010, 0.0025]))

    initial_capital = float(data.get("initial_capital", 10000.0))
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital}")
    cost_rate = float(data["cost_rate"])
    if cost_rate < 0:
        raise ValueError(f"cost_rate cannot be negative, got {cost_rate}")

    data_paths = data.get("data_paths", {})
    raw_dir = Path(data_paths.get("raw_dir", "data/raw"))
    processed_dir = Path(data_paths.get("processed_dir", "data/processed"))
    artifacts_dir = Path(data_paths.get("artifacts_dir", "artifacts"))

    return Settings(
        tickers=tickers,
        acquisition_start=str(data["acquisition_start"]),
        source=str(data["source"]),
        target_horizon=int(data["target_horizon"]),
        target_threshold=float(data["target_threshold"]),
        weights_growth=growth,
        weights_defensive=defensive,
        weights_fixed_baseline=fixed_baseline,
        candidate_thresholds=candidate_thresholds,
        cost_rate=cost_rate,
        cost_sensitivities=cost_sensitivities,
        random_seed=int(data["random_seed"]),
        initial_capital=initial_capital,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
    )
