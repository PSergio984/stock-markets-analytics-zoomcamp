# Replay Remediation Triage & Verification Report

**Date:** 2026-11-18  
**Worktree:** `ai-risk-radar-build`  
**Base Commit:** `2109e056a28989bcfa38dbd5e714ffd787c27da5`  
**Target Specification:** Independent Review Remediation (`.scratch/reviews/replay-agy.md`)  
**Status:** All 7 Findings Remediated, 100% Verified, Clean Test Suite (51 passed)

---

## 1. Executive Summary

An independent review (`.scratch/reviews/replay-agy.md`) identified 7 findings (2 High, 3 Medium, 2 Low) in the initial holdout replay implementation. All 7 findings have been remediated in full with source-backed immutable provenance, trajectory-level mathematical equivalence, strict pre-simulation boundaries, and complete test verification.

Zero modifications were made to frozen baseline artifacts. The test suite has expanded from 45 to 51 tests, all passing, with zero Ruff linter errors.

---

## 2. Detailed Finding-by-Finding Remediation Matrix

| Finding ID & Severity | Root Cause | Remediation Applied | Verification Proof |
| :--- | :--- | :--- | :--- |
| **Finding 1 [HIGH]**<br>Actual historical reproduction divergence | Release 1.0.0 `test_evaluation.test_metrics.total_return` was 0.55290498 (55.29%) computed from post-funding capital ($9,990.01), whereas modern pipeline measured net initial capital ($10,000.00 = 55.14%), diverging by 15.51 bps. | Implemented trajectory-level equity comparison supporting versioned metric conventions (`legacy_post_funding` vs `net_initial_capital`). When evaluated against historical release 1.0.0, legacy post-funding return reproduces with `diff = 0.0`. Dual returns and funding impact (`15.51 bps`) are explicitly reported. | `tests/test_remediation_findings.py::test_finding_1_and_2_actual_historical_release_reproduction` PASSED. `abs(legacy_return - orig_total_return) == 0.0`. |
| **Finding 2 [MEDIUM]**<br>Tests only verified synthetic self-replay | Tests only tested self-replay where a new release verified itself, never testing reproduction of the actual historical release artifacts. | Created an isolated test fixture that copies the real frozen `release.json`, `random_forest.joblib`, and data snapshot to `tmp_path`, executes `run_final_test()`, verifies `reproduction_test_report.json`, and asserts that original files remain byte-for-byte untouched. Authored deterministic legacy fixtures for normal CI. | `tests/test_remediation_findings.py::test_finding_1_and_2_actual_historical_release_reproduction` PASSED. |
| **Finding 3 [HIGH]**<br>Inferring historical policy from mutable config | Historical releases without an embedded `policy_hash` inferred policy from whatever was in `config.json` at execution time, allowing config tampering to simulate unvalidated parameters. | Created `ai_risk_radar/provenance.py` registering immutable `HistoricalReleaseIdentity` for Release 1.0.0 tied to exact model SHA-256 (`f805f206...`) and data SHA-256 (`b476eaff...`). Explicitly checks all committed parameters (`cost_rate=0.001`, weights, seeds) and fails BEFORE simulation starts. | `tests/test_remediation_findings.py::test_finding_3_pre_simulation_sentinel_policy_tampering` patches `simulate()` with a runtime sentinel; verified that modifying `cost_rate` raises `ValueError` before `simulate()` is called. |
| **Finding 4 [MEDIUM]**<br>Incomplete boundary date validation | Boundary checks only validated `test_start`, `test_end`, and `cutoff`, omitting the preceding 6 partition dates and execution conventions. | Pre-simulation verification now validates all 9 partition keys in `expected_boundaries` (`train_start`, `train_end`, `calibration_start`, `calibration_end`, `validation_start`, `validation_end`, `test_start`, `test_end`, `cutoff`) plus execution conventions (`target_horizon=20`, `target_threshold=-0.05`, `random_seed=42`). | `tests/test_remediation_findings.py::test_finding_4_pre_simulation_sentinel_boundary_tampering` and `test_finding_4_pre_simulation_convention_tampering` PASSED. |
| **Finding 5 [MEDIUM]**<br>Silent holdout reset in `run_research()` | `run_research()` unconditionally wrote a fresh `release.json` with `holdout_status = "untouched"`, silently erasing records of consumed test evaluations. | `run_research()` now detects existing holdout consumption, preserves `previous_holdout_consumption` in the new release manifest, and flags `holdout_status = "subsequent_research_holdout_previously_consumed"`. | `tests/test_remediation_findings.py::test_finding_5_holdout_consumption_preservation_across_research` PASSED. |
| **Finding 6 [LOW]**<br>Dashboard failed to load reproduction or distinguish returns | `app.py` never loaded `reproduction_test_report.json` if `final_test_report.json` existed, and presented only a single return metric without reproduction badges. | Updated `load_report_data()` in `app.py` to check and merge reproduction report metadata. Added `✅ Holdout Reproduction: Verified Match` banner and separate metric cards for "Net Initial-Capital Return" (55.1%), "Original Post-Funding Return" (55.3%), and "Initial Setup Cost Impact" (15.51 bps). Provenance caption indicates reproduction status. | `tests/test_remediation_findings.py::test_finding_6_dashboard_loads_reproduction_and_views_format_dual_returns` PASSED. |
| **Finding 7 [LOW]**<br>README documented failing reproduction command | `README.md` Section 5.2 documented running `final-test` without explaining reproduction mode, leading users to believe it would overwrite `final_test_report.json`. | Updated Section 5.2 in `README.md` to document both First-Time Evaluation and Reproduction Mode, explaining that executing on an evaluated release writes `reproduction_test_report.json` while keeping `release.json` immutable. | Verified in `README.md`. |

---

## 3. Cryptographic Immutability Audit

Verification confirmed zero byte-level modifications to any frozen baseline artifacts:

| Artifact | Expected Baseline SHA-256 | Verified Current SHA-256 | Immutability Status |
| :--- | :---: | :---: | :---: |
| `artifacts/release.json` | `1f4751a7108874364533b55f38a5ab1fdd3e26b0689dfbd502319e29098319b9` | `1f4751a7108874364533b55f38a5ab1fdd3e26b0689dfbd502319e29098319b9` | **100% MATCH (Immutable)** |
| `artifacts/final_test_report.json` | `6603ab4c7204bd38b2d179ee67bf17760788240e11f4fc61b267d3c3705d0901` | `6603ab4c7204bd38b2d179ee67bf17760788240e11f4fc61b267d3c3705d0901` | **100% MATCH (Immutable)** |
| `artifacts/models/random_forest.joblib` | `f805f206803067139f8c52099273bf9e80ca53ec64b10aed270daabebbcfe113` | `f805f206803067139f8c52099273bf9e80ca53ec64b10aed270daabebbcfe113` | **100% MATCH (Immutable)** |
| `data/raw/snapshot/manifest.json` | `b476eaff931e68f4ece308cd64e860928860925a6a59358269c29fbeb4df4974` | `b476eaff931e68f4ece308cd64e860928860925a6a59358269c29fbeb4df4974` | **100% MATCH (Immutable)** |
| `artifacts/corrected_holdout_evaluation.json` | `ac961b6b790897fd1b57f501ba43a361eade013dada886a3e65c846cf301857d` | `ac961b6b790897fd1b57f501ba43a361eade013dada886a3e65c846cf301857d` | **100% MATCH (Immutable)** |

---

## 4. Test Suite and Static Analysis Audit

- **Pytest:** 51 passed in 92.64s (`docs/tests.log`).
- **Ruff:** All checks passed with 0 errors (`ruff check .`).
- **Code Coverage:** All 7 findings covered by automated unit & integration tests in `tests/test_remediation_findings.py`.
- **Pre-simulation Sentinel Proof:** Verified that tampering with parameters or partition dates raises `ValueError` before `simulate()` is ever reached.
