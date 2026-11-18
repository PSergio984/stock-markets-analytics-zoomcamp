"""Source-backed provenance registry and migration identity for historical releases."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HistoricalReleaseIdentity:
    """Immutable provenance record tied to exact historical release hashes and committed parameters."""

    release_version: str
    origin_commit: str
    metric_convention: str
    model_hash: str
    data_hash: str
    policy_hash: str
    policy_parameters: dict[str, Any]
    expected_boundaries: dict[str, str]


# Exact SHA-256 digests and committed parameters from historical release 1.0.0 (commit 290cba6)
# release.json: 1f4751a7108874364533b55f38a5ab1fdd3e26b0689dfbd502319e29098319b9
# final_test_report.json: 6603ab4c7204bd38b2d179ee67bf17760788240e11f4fc61b267d3c3705d0901
# random_forest.joblib: f805f206803067139f8c52099273bf9e80ca53ec64b10aed270daabebbcfe113
# manifest.json: b476eaff931e68f4ece308cd64e860928860925a6a59358269c29fbeb4df4974
HISTORICAL_RELEASE_1_0_0 = HistoricalReleaseIdentity(
    release_version="1.0.0",
    origin_commit="290cba6",
    metric_convention="legacy_post_funding",
    model_hash="f805f206803067139f8c52099273bf9e80ca53ec64b10aed270daabebbcfe113",
    data_hash="b476eaff931e68f4ece308cd64e860928860925a6a59358269c29fbeb4df4974",
    # Computed policy hash from original committed config (threshold 0.50, cost_rate 0.001, canonical weights)
    policy_hash="d3b5aeb482ab316a6b8850afab91cf9e12b61b7e4030139aeef3f1c2f540e212",
    policy_parameters={
        "selected_threshold": 0.50,
        "selected_strategy": "threshold_0.50",
        "tickers": ("SOXX", "XLK", "SPY", "SHY"),
        "weights_growth": (0.35, 0.35, 0.20, 0.10),
        "weights_defensive": (0.10, 0.15, 0.35, 0.40),
        "weights_fixed_baseline": (0.20, 0.20, 0.20, 0.40),
        "candidate_thresholds": (0.20, 0.35, 0.50),
        "cost_rate": 0.001,
        "cost_sensitivities": (0.0005, 0.0010, 0.0025),
        "random_seed": 42,
        "initial_capital": 10000.0,
        "target_horizon": 20,
        "target_threshold": -0.05,
    },
    expected_boundaries={
        "train_start": "2016-01-04",
        "train_end": "2021-11-18",
        "calibration_start": "2021-11-18",
        "calibration_end": "2022-11-18",
        "validation_start": "2022-11-18",
        "validation_end": "2024-11-18",
        "test_start": "2024-11-18",
        "test_end": "2026-11-16",
        "cutoff": "2026-11-16",
    },
)

# Registry keyed by (model_hash, data_hash)
HISTORICAL_PROVENANCE_REGISTRY: dict[tuple[str, str], HistoricalReleaseIdentity] = {
    (
        HISTORICAL_RELEASE_1_0_0.model_hash,
        HISTORICAL_RELEASE_1_0_0.data_hash,
    ): HISTORICAL_RELEASE_1_0_0,
}


def lookup_historical_provenance(model_hash: str, data_hash: str) -> HistoricalReleaseIdentity | None:
    """Lookup source-backed migration identity for historical releases lacking embedded policy hashes."""
    return HISTORICAL_PROVENANCE_REGISTRY.get((model_hash, data_hash))
