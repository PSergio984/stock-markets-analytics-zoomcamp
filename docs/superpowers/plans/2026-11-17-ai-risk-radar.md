# AI Investment Risk Radar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an affordable, reproducible course project forecasting AI-sensitive downside risk, evaluating defensive allocations, and presenting real results in Compact / Detailed views.

**Architecture:** Build one independent Python project beneath the course repository. A shared pipeline produces versioned data, model and report artifacts; a thin Streamlit application displays those same artifacts. Historical research, untouched-test evaluation and latest inference are separate operations with explicit data/model identities.

**Tech Stack:** Python, pandas/numpy, yfinance, scikit-learn/joblib, exchange_calendars, matplotlib, Streamlit, Jupyter/nbconvert, pytest and Ruff. Resolve and pin compatible versions in an isolated project environment at execution time; do not modify the user's global environment.

**Spec:** [Approved project design](../specs/2026-11-17-ai-risk-radar-design.md). Executors must read both documents, the [rubric](../../../projects/README.md), and the [research caveats](../../research/2026-11-17-final-project-shortlist.md).

## Global Constraints

- Work inside `final-project/ai-risk-radar/`; teaching modules and approved prototypes are read-only references. Do not create this project until plan handoff is accepted.
- Fixed universe: SOXX, XLK, SPY, SHY; reference target basket is 50% SOXX / 50% XLK, held through each label horizon.
- Target: next 20-trading-day basket total return `<= -0.05`; risk is end-of-horizon loss probability, not intrahorizon drawdown.
- Growth weights: `0.35/0.35/0.20/0.10`; defensive: `0.10/0.15/0.35/0.40`; fixed baseline: `0.20/0.20/0.20/0.40`, in the universe order above.
- Weekly signals; execution at the following trading session's close; costs `0.001` per dollar bought or sold, with sensitivity `0.0005/0.0025`, including initial funding.
- Final 24 months untouched test, preceding 24 validation, preceding 12 calibration, earlier training of at least four years; purge label starts whose 20-session outcomes reach the next partition.
- Candidate risk thresholds `0.20/0.35/0.50`; lowest validation drawdown subject to CAGR at least the fixed baseline; ties favor CAGR then lower turnover. No qualifying candidate means documented failure and fixed-baseline fallback.
- Free local core; no paid services, live trades, leverage, shorts, accounts, cloud deployment, news scraping or autoresearch loop. FRED/ALFRED excluded; Treasury/OFR extensions are conditional on availability evidence.
- No datasets, cached prices, model binaries or generated reports committed to Git. Fail visibly; no sample-data fallback in the real pipeline.
- Compact defaults on; Detailed displays the same report. No financial improvement claims before genuine evaluation.
- Commit checkpoints below describe logical change boundaries. Commit/push only when the user's session authorization permits; this planning task makes no commits.

## Review Focus

- An ETF missing one otherwise valid exchange session must stop artifact publication; the program must not silently bridge a multi-day return as a one-day return. Owned by Task 1.
- A 20-session label ending exactly at a split boundary belongs outside the earlier partition. Owned by Task 2.
- A calibration period with one class must disable calibrated-probability presentation and signal-based trading, with an explicit fallback reason. Owned by Tasks 3 and 5.
- A holiday-shortened week must signal on its actual last session, trade on the next session, and retain old exposure until then. Owned by Task 4.
- Interrupted refresh or a missing report must preserve the previous valid dated report or show an explicit unavailable state in both views. Owned by Tasks 5 and 6.

## File and artifact map

All paths below are relative to `final-project/ai-risk-radar/` unless prefixed otherwise.

| Files | Responsibility |
|---|---|
| `pyproject.toml`, `requirements.in`, `requirements.txt`, `.gitignore`, `config.json` | Project entry point, pinned environment, narrow generated-file exclusions and approved settings |
| `ai_risk_radar/contracts.py`, `config.py`, `__init__.py` | Typed shared records and validated settings |
| `ai_risk_radar/data.py`, `features.py`, `splits.py` | Price snapshot/manifest, causal feature/target table, purged chronological partitions |
| `ai_risk_radar/models.py`, `simulation.py`, `metrics.py` | Training/calibration, delayed allocation execution, measured outcomes |
| `ai_risk_radar/reporting.py`, `pipeline.py`, `__main__.py` | Honest report schema, atomic artifacts, separate research/final-test/latest commands |
| `app.py`, `ai_risk_radar/views.py`, `notebooks/research.ipynb`, `notebooks/end_to_end.ipynb` | Two-mode presentation and notebooks using shared functions |
| `tests/test_*.py`, `README.md`, `docs/qa.md` | Financial correctness tests, reproducible commands, review evidence and rubric mapping |

Ignored local directories: `data/raw/`, `data/processed/`, `artifacts/`, `.venv/`, caches and rendered notebook outputs. Committed unit fixtures must be tiny hand-authored test inputs, clearly unrelated to research results.

## Shared interfaces

Define these typed dataclasses in `contracts.py` as their owning task needs them. Pandas indices are unique, ascending, timezone-normalized exchange session dates.

- `Settings`: tickers, acquisition start, source, target horizon/threshold, weight vectors, risk cutoffs, costs, seed and local paths. Use `tuple[str, ...]`, `tuple[float, ...]`, `int`, `float`, `Path`; validate weights, order and permitted values against config.
- `DataBundle(adjusted_close: pd.DataFrame, volume: pd.DataFrame, sessions: pd.DatetimeIndex, manifest: dict[str, object])`. Raw provider payload and adjustment evidence stay in the snapshot directory.
- `FeatureSet(frame: pd.DataFrame, feature_columns: tuple[str, ...], target_columns: tuple[str, ...])`; columns include `r20`, nullable `downside20`, `label_end` and `is_feature_complete`.
- `SplitSet(train: pd.DatetimeIndex, calibration: pd.DatetimeIndex, validation: pd.DatetimeIndex, test: pd.DatetimeIndex, boundaries: dict[str, str])`.
- `ModelBundle(estimator: object, calibrator: object | None, feature_columns: tuple[str, ...], metadata: dict[str, object])`; metadata includes identities, fit-session boundaries, class counts and `probability_status`.
- `BacktestResult(daily: pd.DataFrame, trades: pd.DataFrame, metrics: dict[str, float | None])`; daily records gross/net returns, equity, drawdown and holdings. Trades record signal date, execution date, before/after weights and charged cost.
- `Report(payload: dict[str, object])`: JSON-serializable, versioned presentation contract. Required sections: `identity`, `data_status`, `forecast`, `allocation`, `performance`, `comparisons`, `diagnostics`, `limitations`. Optional metrics use null plus a reason, never NaN/Infinity.

### Task 1: Establish a real, reproducible price snapshot
Source: [design: Data and availability], [design: Verification and builder discretion], [DECISIONS.md: Technical decisions captured by full-design approval]
Intent: Prevent silent data corruption, misaligned sessions across assets, double-counting dividend adjustments, unverified network reliance, and unrepeatable environment setup.

**Files:** Create project/environment files, `contracts.py`, `config.py`, `data.py`, `tests/test_data.py`, and initial README setup/data instructions.

**Interfaces:** Produce `load_settings(path: Path) -> Settings`, `fetch_prices(settings: Settings, as_of: pd.Timestamp) -> DataBundle`, `load_snapshot(path: Path) -> DataBundle`, and `save_snapshot(bundle: DataBundle, path: Path) -> Path`. `save_snapshot` returns the hashed manifest path.

- [ ] Write deterministic tests `test_missing_exchange_session_rejected`, `test_duplicate_or_unsorted_sessions_rejected`, `test_adjustments_do_not_double_count_dividends`, `test_snapshot_hash_mismatch_rejected`, and `test_weight_order_and_sum_validated`. Assert a missing SOXX session while SPY trades raises a dated error; a toy split/dividend preserves the documented total-return calculation; corrupt cached bytes fail verification.
  - Source: [design: Data and availability], [design: Verification and builder discretion], [DECISIONS.md: Still to verify during implementation]
  - Intent: Prevent silent exchange session mismatches across assets (e.g. SOXX missing when SPY trades), dividend distortion from double-counting, unsorted/duplicate timestamps, invalid portfolio weight vectors, and silent cache byte corruption.
- [ ] Create a project-only virtual environment, resolve compatible dependencies, pin the resolved environment and install this package editable. Run `python -m pytest tests/test_data.py -q`; verify the tests fail because ingestion/contracts are not implemented.
  - Source: [design: Delivery and reproducibility], [DECISIONS.md: Technical decisions captured by full-design approval]
  - Intent: Prevent dependency pollution of global system packages, ensure reproducible execution under isolated virtual environment, and verify strict RED TDD gate before code implementation.
- [ ] Implement the interfaces using yfinance's explicitly recorded adjustment settings and an XNYS session calendar. Keep raw action evidence separate from adjusted series. Establish a completed-session cutoff and reject incomplete sessions/nonpositive prices; do not fill missing asset prices from future sessions. Cached mode performs no network calls.
  - Source: [design: Data and availability], [design: Review Focus]
  - Intent: Prevent silent bridging of multi-day returns as single-day returns, prevent future-session lookahead filling, prevent nonpositive prices, and ensure zero-network offline reproducibility in cached mode.
- [ ] Perform the opt-in real data feasibility check for the four ETFs from 2016 onward. Save source/version, adjustment settings, requested/actual coverage, sessions, missingness and hashes; compare a documented split/dividend example to returned prices. Provider/auth/history failures stop this task. An alternative source requires documented use/access and explicit source selection before resuming.
  - Source: [design: Proposed scope and defaults], [design: Data and availability], [DECISIONS.md: Still to verify during implementation]
  - Intent: Prevent building on unavailable, truncated, or unadjusted price data; ensure provider and ETF coverage meet the 4-year training plus calibration/validation/test requirements without synthetic fallback.
- [ ] Run the data tests, `python -m ruff check .`, and a cache reload with identical hashes. Evidence: real usable history sufficient for the required partitions, no hidden holes, and a project-only pinned environment. Checkpoint: `feat: add reproducible ETF data snapshots`.
  - Source: [design: Delivery and reproducibility], [design: Verification and builder discretion]
  - Intent: Prevent regressions, lint defects, and byte hash drifts between original fetch and subsequent cached runs.

### Task 2: Build causal features, exact labels and purged splits
Source: [design: Prediction target and features], [design: Models and chronological evaluation], [DECISIONS.md: Confirmed by the user]
Intent: Prevent lookahead bias in feature engineering, label leakage across chronological splits, and incorrect loss definitions.

**Files:** Create `features.py`, `splits.py`, `tests/test_features.py`, `tests/test_splits.py`; extend `contracts.py` and `config.json`.

**Interfaces:** Consume `DataBundle` and `Settings`. Produce `build_features(bundle: DataBundle, settings: Settings) -> FeatureSet` and `make_splits(features: FeatureSet, settings: Settings) -> SplitSet`.

- [ ] Write `test_target_is_equal_weight_buy_and_hold`: over 20 sessions SOXX grows 20% and XLK loses 30%; assert `r20 == -0.05` within tolerance and `downside20 == 1`. Add `test_unknown_tail_labels_preserve_inference_rows` asserting final 20 sessions have unknown outcomes but complete trailing features; `test_future_price_change_cannot_change_past_features`; and `test_label_end_at_boundary_is_purged`.
  - Source: [design: Prediction target and features], [design: Models and chronological evaluation], [DECISIONS.md: Confirmed by the user]
  - Intent: Prevent formula deviations in 50/50 SOXX/XLK 20-session return and downside binary threshold (-0.05), prevent truncation of latest unlabeled inference sessions, prevent future data leaking into historical features, and ensure boundary-crossing labels are purged.
- [ ] Run `python -m pytest tests/test_features.py tests/test_splits.py -q`; expect missing feature/split interfaces to fail.
  - Source: [executing-plans / test-driven-development skill], [AGY-IMPLEMENTATION-BRIEF.md: Work to perform]
  - Intent: Prevent writing untested feature/split code by verifying legitimate RED TDD failures for missing interfaces.
- [ ] Implement price-derived features using lookbacks 1/5/20/63 sessions, volatility 20/63, drawdown 63, SOXX/XLK/SPY relative strength and rolling correlations. Define `NUMERIC`, `TO_PREDICT` and optional date-category groups explicitly. Never include `r20`, labels or `label_end` in model inputs. Start with price-only features; conditional macro extensions remain separate experiments.
  - Source: [design: Data and availability], [design: Prediction target and features]
  - Intent: Prevent future information leakage into price indicators, prevent target labels or future returns from leaking into input feature matrices, and strictly define column groups.
- [ ] Anchor partition dates to the frozen last completed session using the approved calendar-month spans. Apply coverage requirements and purge based on actual `label_end`, not an approximate calendar-day gap. Save exact boundaries and feature-column order in the research manifest before outcome analysis.
  - Source: [design: Models and chronological evaluation], [design: Review Focus]
  - Intent: Prevent calendar-day approximation bugs by using exact trading session dates for 24m test, 24m validation, 12m calibration, and >=4y train; prevent horizon leakage by purging labels whose 20-session outcome spans across boundaries.
- [ ] Run the task tests and Ruff. Verify changing only test prices cannot alter earlier features, split boundaries or training preprocessing inputs. Checkpoint: `feat: add causal features and purged downside labels`.
  - Source: [design: Delivery and reproducibility], [design: Verification and builder discretion]
  - Intent: Verify strict temporal isolation (modifying test data does not alter training features or split indices) and code quality standards.

### Task 3: Train and check interpretable risk estimates
Source: [design: Models and chronological evaluation], [design: Review Focus], [DECISIONS.md: Technical decisions captured by full-design approval]
Intent: Prevent information leakage during training/calibration, overfitted model complexity, and false probability claims under single-class or inadequate event support.

**Files:** Create `models.py`, `tests/test_models.py`; extend `contracts.py`.

**Interfaces:** Consume `FeatureSet`, `SplitSet`, `Settings`. Produce `fit_models(features: FeatureSet, splits: SplitSet, settings: Settings) -> dict[str, ModelBundle]`, `predict_risk(bundle: ModelBundle, frame: pd.DataFrame) -> pd.Series`, and `classification_report(y: pd.Series, probability: pd.Series) -> dict[str, object]`.

- [ ] Write `test_fit_never_sees_validation_or_test`, `test_calibrator_uses_only_calibration_sessions`, `test_constant_predictor_uses_training_prevalence`, `test_one_class_calibration_disables_probability`, and `test_serialized_model_predictions_match`. Assert changing holdout labels cannot change model parameters; invalid calibration produces `probability_status='unavailable'` and a reason, not a fake percentage.
  - Source: [design: Models and chronological evaluation], [design: Review Focus], [DECISIONS.md: Still to verify during implementation]
  - Intent: Prevent training models on validation/test data, calibrating on non-calibration partitions, claiming false calibration when only one class is present, and serialization prediction mismatches.
- [ ] Run `python -m pytest tests/test_models.py -q`; expect missing model interfaces to fail.
  - Source: [executing-plans / test-driven-development skill], [AGY-IMPLEMENTATION-BRIEF.md: Work to perform]
  - Intent: Verify that model and evaluation assertions produce genuine RED failures prior to implementation.
- [ ] Fit a constant prevalence baseline, Decision Tree depths 2/3 and Random Forest with 200 trees, depths 3/5 and minimum leaf sizes 20/50, random seed 42. Compare nonconstant candidates with three expanding, purged training-only folds using Brier score. Select parameters without validation/test data; then fit sigmoid calibration on the calibration block only.
  - Source: [design: Models and chronological evaluation]
  - Intent: Prevent hyperparameter selection using test/validation data by restricting tuning to 3 expanding purged training-only folds, and fit sigmoid calibration strictly on calibration data.
- [ ] Define minimum calibration support as at least 20 labeled sessions per class and at least two positive label starts separated by 20 sessions. Report class/event counts on a deterministic non-overlapping 20-session subsample separately; meeting this minimum is not proof of reliable calibration. If support is inadequate, preserve uncalibrated diagnostics but disable probability-based allocation and use the declared fixed baseline. Do not lower the criterion after reading test outcomes. This statistical implementation choice may make the model inconclusive, which is an acceptable finding.
  - Source: [design: Models and chronological evaluation], [design: Review Focus]
  - Intent: Prevent displaying misleading or uncalibrated probabilities as real risk percentages when event counts are statistically insufficient; fall back cleanly to uncalibrated diagnostics and fixed baseline.
- [ ] Report validation Brier score, event prevalence, precision/recall at candidate thresholds and reliability bins, including non-overlapping evaluation. Run model tests and Ruff; verify persisted estimator/calibrator identities and identical predictions on reload. Checkpoint: `feat: train and calibrate downside risk models`.
  - Source: [design: Models and chronological evaluation], [design: Delivery and reproducibility]
  - Intent: Prevent reporting biased performance metrics by evaluating non-overlapping blocks alongside daily series, ensuring model persistence and reproducibility.

### Task 4: Simulate exposure changes without timing or cost shortcuts
Source: [design: Trading experiment], [design: Review Focus], [DECISIONS.md: Confirmed by the user]
Intent: Prevent execution timing shortcuts, uncharged transaction costs, unrealistic instant fills, and ungrounded strategy selection.

**Files:** Create `simulation.py`, `metrics.py`, `tests/test_simulation.py`, `tests/test_metrics.py`.

**Interfaces:** Produce `weekly_targets(risk: pd.Series, bundle: DataBundle, settings: Settings, threshold: float) -> pd.DataFrame`, `simulate(prices: pd.DataFrame, targets: pd.DataFrame, settings: Settings, cost_rate: float) -> BacktestResult`, `benchmark_targets(bundle: DataBundle, settings: Settings) -> dict[str, pd.DataFrame]`, and `select_strategy(candidates: dict[str, BacktestResult], baseline: BacktestResult) -> str | None`.

- [ ] Write `test_holiday_week_executes_next_session_close`, `test_old_weights_earn_until_execution`, `test_buy_and_hold_weights_drift`, `test_costs_charge_each_buy_and_sell`, `test_initial_funding_cost_is_charged`, and `test_no_qualifying_candidate_returns_none`. For a full SOXX-to-SHY transfer with zero market return at rate 0.001, assert net wealth factor `(1-c)/(1+c)`; initial all-cash funding gives `1/(1+c)`. Also assert no trade occurs on an absent session.
  - Source: [design: Trading experiment], [design: Review Focus]
  - Intent: Prevent timing lookahead on holiday weeks, prevent applying new weights before execution session close, prevent zero-cost assumptions, and verify conservation of equity under buy/sell fees.
- [ ] Run `python -m pytest tests/test_simulation.py tests/test_metrics.py -q`; expect missing simulation/metric functions to fail.
  - Source: [executing-plans / test-driven-development skill], [AGY-IMPLEMENTATION-BRIEF.md: Work to perform]
  - Intent: Verify genuine RED TDD failures for simulation and financial metrics prior to implementation.
- [ ] Implement aligned daily portfolio returns with drifting holdings between rebalances. Let `Vpre` be equity before execution, `h` pretrade dollar holdings, `w` target weights and `c` cost rate. Solve `Vpost = Vpre - c * sum(abs(w*Vpost - h))` so posttrade holdings and transaction costs conserve equity, including funding. Apply new weights only after execution close. Use a positive reference capital of USD 10,000 for display; normalized curves and returns drive comparison.
  - Source: [design: Trading experiment]
  - Intent: Prevent unrealistic constant-weight assumptions between weekly rebalances, accurately charge transaction costs on every buy and sell (including initial funding), and conserve portfolio equity.
- [ ] Implement growth buy-and-hold, fixed defensive weekly and 63-session momentum policies on identical dates and conventions. Select threshold using the approved validation rule; all candidate results remain visible. Compute CAGR, peak-to-trough drawdown, volatility, turnover and defensive occupancy; report annualized Sharpe with explicitly zero risk-free assumption, and null for zero volatility or insufficient observations.
  - Source: [design: Trading experiment], [DECISIONS.md: Technical decisions captured by full-design approval]
  - Intent: Prevent strategy bias by benchmarking against standard policies under identical rules, select candidate threshold objectively (lowest validation drawdown subject to CAGR >= fixed baseline), and report honest metrics (Sharpe with rf=0).
- [ ] Run task tests, cost sensitivities 5/10/25 basis points and Ruff. Check that target labels never enter the return engine, all weights sum to one, and costs cannot increase wealth. Checkpoint: `feat: add delayed allocation simulation and baselines`.
  - Source: [design: Trading experiment], [design: Verification and builder discretion]
  - Intent: Prevent hidden trading cost fragility, verify no label data leaks into portfolio weights, and ensure weight vector sum equals 1.

### Task 5: Publish research, untouched-test and latest reports safely
Source: [design: Delivery and reproducibility], [design: Review Focus], [DECISIONS.md: Technical decisions captured by full-design approval]
Intent: Prevent test set contamination, partial or corrupted report updates, invalid JSON structures, and unauthorized model retraining during production inference.

**Files:** Create `reporting.py`, `pipeline.py`, `__main__.py`, `tests/test_pipeline.py`, `tests/test_reporting.py`; extend `contracts.py`.

**Interfaces:** Produce `run_research(settings: Settings, data_mode: str) -> Path`, `run_final_test(release_manifest: Path) -> Path`, `run_latest(release_manifest: Path, data_mode: str) -> Path`, `build_report(results: dict[str, BacktestResult], forecast: pd.Series, metadata: dict[str, object]) -> Report`, and `load_report(path: Path) -> Report`.

- [ ] Write `test_research_does_not_evaluate_test_outcomes`, `test_release_hash_change_blocks_test_reuse`, `test_latest_does_not_fit_models`, `test_incomplete_refresh_preserves_previous_report`, `test_missing_report_is_explicit`, and `test_unavailable_probability_uses_fixed_baseline`. Assert serialized reports contain neither nonfinite JSON nor illustrative prototype constants.
  - Source: [design: Delivery and reproducibility], [design: Review Focus]
  - Intent: Prevent peeking at test results during research, prevent re-evaluating test on altered releases, prevent retraining during latest inference, and ensure graceful handling of missing or partial reports.
- [ ] Run `python -m pytest tests/test_pipeline.py tests/test_reporting.py -q`; expect missing pipeline/report interfaces to fail.
  - Source: [executing-plans / test-driven-development skill], [AGY-IMPLEMENTATION-BRIEF.md: Work to perform]
  - Intent: Verify RED TDD test failures for pipeline orchestration and atomic report serialization.
- [ ] Implement commands from the project directory: `python -m ai_risk_radar research --config config.json --data-mode refresh`, `python -m ai_risk_radar final-test --release artifacts/release.json`, and `python -m ai_risk_radar latest --release artifacts/release.json --data-mode cache`. `refresh` and `cache` are the only data modes. Research fits/selects using earlier blocks and freezes data/model/policy hashes; final-test evaluates that frozen release once, allowing identical-hash reproduction but refusing a changed release masquerading as the same holdout experiment.
  - Source: [design: Delivery and reproducibility], [design: Models and chronological evaluation]
  - Intent: Prevent conflation of research, frozen holdout evaluation, and latest inference by enforcing explicit command-line boundaries and hash verification.
- [ ] Latest mode loads the frozen demonstration model, preserves feature-complete unlabeled dates and records model age; it performs no retraining. Publish JSON and table artifacts atomically and only after validation, with `identity`, `data_status`, `forecast`, `allocation`, `performance`, `comparisons`, `diagnostics`, `limitations`. Distinguish observed test performance, validation selection and latest future forecast. Report data staleness by completed exchange sessions and failed refresh state. Weekly suggestions include no-change outcomes.
  - Source: [design: Delivery and reproducibility], [design: User experience]
  - Intent: Prevent silent model retraining in production, track data freshness and staleness explicitly, and provide clear actionable allocation signals.
- [ ] Run task tests, Ruff and one cache-only pipeline integration using a tiny deterministic unit fixture. Then run real research and freeze the release; evaluate the final holdout only after parameters and reporting are fixed. Evidence: commands, hashes, candidate table and honest outcome. Checkpoint: `feat: publish versioned research and latest risk reports`.
  - Source: [design: Delivery and reproducibility], [design: Verification and builder discretion]
  - Intent: Ensure end-to-end pipeline reproducibility on frozen fixtures before consuming the single-use holdout evaluation.

### Task 6: Connect the approved two-mode dashboard to real artifacts
Source: [design: User experience], [DECISIONS.md: Confirmed by the user], [docs/prototypes/ai-risk-radar/README.md]
Intent: Prevent data divergence between views, UI calculations altering underlying metrics, deceptive prototype placeholder displays, or unhandled stale data states.

**Files:** Create `app.py`, `views.py`, `tests/test_views.py`, `tests/test_app.py`.

**Interfaces:** Consume `Report` via `load_report`; produce `compact_view(report: Report) -> dict[str, object]`, `detailed_view(report: Report) -> dict[str, object]`, and `render_dashboard(report_path: Path) -> None`. Views format data only; they never fit models, fetch prices or simulate trades.

- [ ] Write `test_modes_share_forecast_and_metrics`, `test_missing_or_stale_report_is_visible`, `test_forecast_label_states_20_sessions_and_5_percent`, `test_missing_calibration_never_shows_probability`, and `test_percentage_points_are_not_relative_percentages`. Use Streamlit AppTest for default Compact selection and switching to Detailed without changing report identity.
  - Source: [design: User experience], [docs/prototypes/ai-risk-radar/README.md]
  - Intent: Prevent discrepancies between Compact and Detailed view calculations, ensure missing/stale reports are surfaced visibly, and ensure honest label and percentage point representations.
- [ ] Run `python -m pytest tests/test_views.py tests/test_app.py -q`; expect missing UI/view functions to fail.
  - Source: [executing-plans / test-driven-development skill], [AGY-IMPLEMENTATION-BRIEF.md: Work to perform]
  - Intent: Verify RED TDD failures for view formatting and dashboard presentation components.
- [ ] Implement `python -m streamlit run app.py`. Compact shows as-of date, probability or its unavailable reason, brief measured-feature explanation, trade-off metrics and suggested research weights. Detailed adds comparisons, calibration, allocation history, feature/correlation charts and failure periods. Use shared formatting and artifact values; discard every fabricated mockup metric.
  - Source: [design: User experience], [design: Proposed scope and defaults]
  - Intent: Provide an interactive, responsive presentation of real pipeline artifacts in both Compact (default) and Detailed modes without recalculating models or fetching prices in the UI.
- [ ] Inspect both modes at desktop and 375px, keyboard selection, missing report and stale last-valid report. Record screenshots and actual viewport widths in `docs/qa.md`; wide tables may scroll within containers. Verify headings distinguish end-return risk from historical maximum drawdown and explain the zero-risk-free Sharpe convention.
  - Source: [design: User experience], [design: Verification and builder discretion]
  - Intent: Prevent mobile layout breaking, horizontal overflow, inaccessible keyboard navigation, and verify correct handling of stale or missing reports.
- [ ] Run UI tests and Ruff. Evidence: real report-driven desktop/mobile views with mode parity and visible unavailable states. Checkpoint: `feat: add compact and detailed risk dashboard`.
  - Source: [design: Delivery and reproducibility], [design: Verification and builder discretion]
  - Intent: Ensure dashboard view logic and rendering adhere to formatting standards and unit verification.

### Task 7: Complete notebooks, automation and submission evidence
Source: [design: Delivery and reproducibility], [projects/README.md], [DECISIONS.md: Confirmed by the user]
Intent: Prevent missing rubric deliverables, unreproducible notebooks, unverified scheduling commands, and unstated financial limitations.

**Files:** Create `notebooks/research.ipynb`, `notebooks/end_to_end.ipynb`, `docs/qa.md`; finish README and project scripts documentation.

**Interfaces:** Notebooks call the functions defined above; they contain no second modeling/simulation implementation. End-to-end notebook selects refresh/cache explicitly and shows download, transform, predict, simulate and latest suggested allocation changes.

- [ ] Build EDA from the frozen research dataset: target distribution, feature descriptions/correlations, missingness, non-overlapping event counts and chronological boundaries. Discuss proxy limits, outcomes that failed, cost sensitivity and the return cost of downside protection. Keep test data out of exploratory model selection.
  - Source: [projects/README.md: Data Transformations + EDA], [design: Delivery and reproducibility]
  - Intent: Prevent exploratory bias into the test set, clearly visualize correlations and event distributions, and document proxy limitations honestly.
- [ ] Build the end-to-end calling notebook and execute a cached copy with `python -m jupyter nbconvert --to notebook --execute notebooks/end_to_end.ipynb --output-dir artifacts/notebooks`. Evidence is successful execution from documented inputs, not checked-in cell screenshots or data blobs.
  - Source: [projects/README.md: Automation], [design: Delivery and reproducibility]
  - Intent: Prevent dead code or workflow divergence between interactive notebooks and library modules by verifying a clean headless execution.
- [ ] Document fresh setup, pinned installation, refresh/cache commands, separate research/final-test/latest operations, dashboard startup, expected outputs, source access and failure recovery. Include a Windows Task Scheduler recipe for daily `latest --data-mode refresh`, working directory and failure logs; provide optional cron equivalent. Inspect Task Scheduler argument quoting and manually run the exact command before claiming automation works. Creating a persistent scheduled task itself needs a later explicit user request.
  - Source: [projects/README.md: Automation], [design: Delivery and reproducibility]
  - Intent: Prevent setup friction, clarify Windows Task Scheduler daily refresh execution without registering unauthorized persistent tasks, and document operational recovery.
- [ ] Complete a rubric table with paths/evidence for all six mandatory areas and actually achieved extras. Record performance honestly; no automatic profitability or grade promise. Include submission instructions and the user's three required peer reviews as external remaining actions.
  - Source: [projects/README.md: Evaluation Criteria], [design: Delivery and reproducibility]
  - Intent: Ensure complete, auditable alignment with every course evaluation criterion and honest documentation of strategy risks and limitations.
- [ ] From the isolated environment run `python -m pytest -q`, `python -m ruff check .`, the cached end-to-end notebook, and a fresh-process dashboard smoke check. Inspect `git status --short` and narrow ignore rules: datasets, models and generated reports must be absent from proposed commits. Checkpoint: `docs: document reproducible final project and evaluation`.
  - Source: [design: Delivery and reproducibility], [design: Verification and builder discretion]
  - Intent: Prevent check-in of dirty binaries, data dumps, or broken tests, ensuring a completely reproducible final project.

## Self-review and handoff

- Spec coverage: Task 1 owns acquisition and environment; Task 2 causal table/labels/splits; Task 3 calibrated ML; Task 4 policy/benchmarks/costs; Task 5 reproducible research/latest outputs; Task 6 the approved UI; Task 7 notebooks/automation/rubric. Optional macro features and deployment are explicitly deferred.
- Interface review: the exact dataclasses/signatures above are shared across tasks; extend a contract in its owner before consumers change. No task reads another task's private scratch files.
- Review Focus: each listed failure has named assertions in its owning task. Data/stale state and insufficient event support propagate through reports into both UI modes.
- Evidence status: this document is a plan, not execution results. No data downloads, dependencies, model fitting, backtests or production code were performed to write it.
- Execution handoff: ask the user to review this plan and select native execution or task-by-task agent implementation with independent review. The earlier Agy delegation covered visual mockups only; it does not assign all future implementation to Agy. Recommend independent review of the financially consequential stages even if one agent implements the work.
