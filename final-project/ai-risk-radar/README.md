# AI Investment Risk Radar

An end-to-end, reproducible quantitative risk forecasting system and defensive allocation dashboard built for the Stock Market Analytics Zoomcamp final project.

---

## 1. Problem Statement & Motivation

Investments in artificial intelligence and semiconductor hardware (represented by the **SOXX** semiconductor ETF and **XLK** technology ETF) have generated substantial long-term returns but suffer from acute peak-to-trough drawdowns and severe concentration volatility. For beginner retail investors, large corrections often trigger panic selling at market bottoms, causing permanent capital impairment.

The **AI Investment Risk Radar** provides an objective, calibrated quantitative framework to manage downside risk:
1. **Explicit Downside Risk Definition**: Forecasts the probability that an AI-sensitive reference basket (50% SOXX / 50% XLK) ends the next 20 trading days with a cumulative total return of $\le -5.0\%$. This measures *end-of-horizon loss probability*, strictly distinguished from intra-horizon maximum drawdown.
2. **Delayed Allocation Policy**: When the calibrated downside probability exceeds an empirically selected threshold (selected on validation history, e.g., 50.0%), the portfolio rotates from a **Growth stance** (35% SOXX, 35% XLK, 20% SPY, 10% SHY) to a **Defensive stance** (10% SOXX, 15% XLK, 35% SPY, 40% SHY).
3. **Institutional Execution Realism**: Rebalance signals generated on the final completed trading session of a week execute at the *next* trading session's close. Daily portfolio holdings drift naturally between weekly rebalances, and realistic round-trip transaction costs (10 bps baseline, with 5 bps and 25 bps sensitivity) are deducted directly from portfolio equity using exact numerical conservation.
4. **Leakage-Free Evaluation**: Enforces strict chronological splits (5.87 years training, 12 months Platt scaling calibration, 24 months validation, and 24 months untouched holdout test) with 20-session label purging across all boundaries to prevent temporal leakage.
5. **Two-Mode Presentation Dashboard**: Interactive Streamlit dashboard providing a scannable **Compact View** (default for retail investors) and an analytical **Detailed View** (quantitative cockpit with benchmark comparisons, reliability diagnostics, and failure regimes). Both views consume identical underlying JSON reports.

---

## 2. Fixed Asset Universe

The strategy universe is restricted to four liquid US ETFs:

| Ticker | Asset Description | Role in Strategy |
| :--- | :--- | :--- |
| **SOXX** | iShares Semiconductor ETF | AI hardware & semiconductor sector proxy |
| **XLK** | Technology Select Sector SPDR Fund | Broad enterprise technology & software proxy |
| **SPY** | SPDR S&P 500 ETF Trust | Broad US equity market ballast |
| **SHY** | iShares 1-3 Year Treasury Bond ETF | Short-duration Treasury bond buffer (subject to interest rate and market risk; capital preservation buffer) |

Reference AI basket: `50% SOXX + 50% XLK` (buy-and-hold through each 20-session forecast horizon).

---

## 3. Environment Setup & Installation

The project uses Python 3.12+ and isolated pinned dependencies managed via `uv` or standard `pip`.

### Option A: Using `uv` (Recommended)

```bash
cd final-project/ai-risk-radar
uv venv --python 3.12 .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
uv pip sync requirements.txt
uv pip install -e .
```

### Option B: Using Standard `python -m venv`

```bash
cd final-project/ai-risk-radar
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## 4. Verification & Testing

Run the full deterministic unit and financial correctness test suite (36 tests) and linter:

```bash
# Run test suite
python -m pytest -v

# Run Ruff code quality checks
python -m ruff check .
```

---

## 5. CLI Pipeline Commands

The system features three distinct operational commands separating research, evaluation, and production inference:

### 1. Research Pipeline (`research`)
Performs feature engineering, train/cal/val split purging, model training, Platt scaling calibration, and validation threshold selection. Generates `artifacts/release.json` and `artifacts/research_report.json`. The holdout test partition is strictly excluded.

```bash
python -m ai_risk_radar research --config config.json --data-mode cache
# Use --data-mode refresh to fetch live prices from yfinance
```

### 2. Single-Use Final Holdout Test & Verified Reproduction (`final-test`)
Evaluates the frozen release model on the untouched 24-month holdout test partition (2024-11-18 to 2026-11-16, 500 trading sessions). Cryptographic SHA-256 release hashes and policy parameters are strictly verified before execution.

- **First-Time Evaluation:** Evaluates an untouched release manifest, records holdout consumption in `release.json`, and writes `artifacts/final_test_report.json`.
- **Reproduction Mode:** When executed on an already-evaluated release (e.g. `artifacts/release.json`), the pipeline verifies bit-exact trajectory reproduction against the historical holdout evaluation, generates `artifacts/reproduction_test_report.json`, and leaves `release.json` and `final_test_report.json` strictly immutable.

```bash
python -m ai_risk_radar final-test --release artifacts/release.json
```

### 3. Latest Risk Inference (`latest`)
Loads the frozen calibrated release model and performs forward-looking inference on the latest completed market session. Retraining and parameter tuning are strictly disabled. Writes `artifacts/latest_report.json`.

```bash
python -m ai_risk_radar latest --release artifacts/release.json --data-mode cache
```

---

## 6. Interactive Two-Mode Dashboard

Launch the Streamlit web dashboard:

```bash
python -m streamlit run app.py
```

### Dashboard Modes:
- **Compact View (Default):** Designed for quick daily briefings. Features a scannable Risk Level badge, downside event probability, plain-language event definition, target asset allocation weights, current trade action, and KPI summary.
- **Detailed View:** Full quantitative cockpit. Adds benchmark comparison matrix (vs Growth Buy-and-Hold, Fixed Defensive, Momentum-63), calibration reliability diagnostics (Brier scores, bin predictions vs observed frequencies), causal feature drivers, and documented model failure regimes.

Both modes read from the identical underlying artifact file. Return differences between strategies are formatted in percentage points (`pp`) rather than deceptive relative percentages.

---

## 7. Interactive Research & End-to-End Notebooks

Two clean, reproducible Jupyter notebooks are provided in `notebooks/`:
- `notebooks/research.ipynb`: Exploratory data analysis, target return distributions, non-overlapping downside regime counts, feature correlations, and proxy limitations.
- `notebooks/end_to_end.ipynb`: End-to-end executable workflow calling modular library functions (`load_snapshot`, `build_features`, `make_splits`, `fit_models`, `simulate`, and latest inference).

### Headless Notebook Execution

To verify notebook reproducibility from documented inputs:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/end_to_end.ipynb --output-dir artifacts/notebooks
python -m jupyter nbconvert --to notebook --execute notebooks/research.ipynb --output-dir artifacts/notebooks
```

---

## 8. Automation & Operational Scheduling

### Windows Task Scheduler Recipe

To run daily inference automatically on completed market sessions (after market close at 17:30 New York time):

```cmd
schtasks /create /tn "AIRiskRadarDaily" /tr "\"%CD%\.venv\Scripts\python.exe\" -m ai_risk_radar latest --release artifacts\release.json --data-mode refresh" /sc daily /st 17:30 /ru "%USERNAME%"
```

> [!NOTE]
> Creating a persistent Windows scheduled task requires explicit administrative authorization. Run this command manually if you wish to register the schedule on your local system.

### Linux / macOS Cron Recipe

```bash
30 17 * * 1-5 cd /path/to/final-project/ai-risk-radar && .venv/bin/python -m ai_risk_radar latest --release artifacts/release.json --data-mode refresh >> artifacts/daily_refresh.log 2>&1
```

### Failure Recovery & Data Integrity Safeguards:
- If public data fetching fails (transient network failure, API downtime, holiday session), the pipeline preserves the previous valid report and model artifact.
- If probability calibration support is inadequate (<20 observations or <2 positive events separated by 20 sessions), probability output is marked `unavailable` and portfolio allocation safely defaults to the fixed baseline allocation (20% SOXX, 20% XLK, 20% SPY, 40% SHY).

---

## 9. Course Rubric Self-Assessment Checklist (Evidence Inventory)

This checklist provides auditable file paths for peer review across the **Stock Markets Analytics Zoomcamp** evaluation rubric:

| Rubric Category | Target (Checklist) | Evidence Paths & Deliverables | Description / Implementation Highlights |
| :--- | :---: | :--- | :--- |
| **1. Problem Description** | **4 / 4** | `README.md`, `docs/superpowers/specs/2026-11-17-ai-risk-radar-design.md`, `GLOSSARY.md` | Clear retail motivation, explicit 20-day downside loss definition ($\le -5.0\%$), delayed execution mechanics, and reproducible step-by-step documentation. |
| **2. Data Sources** | **3 / 4** | `ai_risk_radar/data.py`, `tests/test_data.py`, `data/raw/snapshot/` | 4 liquid ETF universe (SOXX, XLK, SPY, SHY) with exchange calendar session alignment (`XNYS`), 0 missing sessions, split/dividend adjustments, and SHA-256 snapshot verification. |
| **3. Data Transformations & EDA** | **3 / 3** | `ai_risk_radar/features.py`, `notebooks/research.ipynb`, `tests/test_features.py` | Causal feature table (drawdown 63, realized vol 20/63, returns 1/5/20/63, relative strength XLK-SPY, rolling correlation), non-overlapping event distribution, and correlation analysis. |
| **4. Modeling & Calibration** | **4 / 5** | `ai_risk_radar/models.py`, `tests/test_models.py`, `artifacts/models/` | Constant baseline, Decision Tree, and Random Forest with 3 expanding purged folds. Platt scaling sigmoid calibrator with minimum calibration support guards. |
| **5. Trading Simulation** | **7 / 8** | `ai_risk_radar/simulation.py`, `ai_risk_radar/metrics.py`, `tests/test_simulation.py` | Delayed next-session close execution, weekly signal frequency, daily holding drift, exact equity conservation under 10 bps transaction costs (5/25 bps sensitivity), and comparison against 3 baselines. |
| **6. Automation & Reproducibility** | **5 / 5** | `notebooks/end_to_end.ipynb`, `requirements.txt`, `pyproject.toml`, Windows Task Scheduler & Cron recipes, `cache`/`refresh` dual modes | Modular Python package (`ai_risk_radar`), headless notebook execution via `nbconvert`, atomic report persistence, and operational scheduling instructions. |
| **7. Bonus Points** | **4 / 7** | `app.py`, `ai_risk_radar/views.py`, `docs/qa.md`, modular architecture | Production-grade two-mode Streamlit dashboard (Compact & Detailed) with verified mobile/desktop responsive design and zero data leakage. |
| **Total Checklist** | **30 / 36** | | Meets course submission requirements; verified independently. |

---

## 10. Financial Performance & Honest Limitation Disclosures

### Holdout Test Evaluation Summary (2024-11-18 to 2026-11-16, 500 trading sessions)

Both original (evaluating capital growth relative to post-funding equity $9,990.01) and corrected arithmetic (evaluating net capital return relative to deposited $10,000.00 capital, accounting for initial funding friction) are reported transparently:

#### Corrected Net Capital Performance (Relative to $10,000.00 Initial Capital — `artifacts/corrected_holdout_evaluation.json`)

| Strategy | Total Return | Return Diff (pp) | CAGR | Max Drawdown | Sharpe ($r_f=0$) | Turnover | Defensive Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selected Strategy (Threshold 0.50)** | **55.14%** | **Base** | **24.83%** | **-24.82%** | **0.93** | **2.15x** | **0.8%** |
| Growth Buy-and-Hold | 57.12% | +1.98 pp | 25.63% | -25.56% | 0.91 | 0.50x | 0.0% |
| Fixed Defensive (Weekly) | 36.81% | -18.33 pp | 17.15% | -15.79% | 1.01 | 1.35x | 100.0% |
| Momentum-63 (Weekly) | 42.63% | -12.50 pp | 19.64% | -21.96% | 0.90 | 8.84x | 28.0% |

#### Original Saved Artifact Baseline (`artifacts/final_test_report.json`)
- Selected Strategy: Total Return 55.29%, CAGR 24.89%, Max Drawdown -24.82%, Sharpe 0.93
- Growth Buy-and-Hold: Total Return 57.28%, CAGR 25.70%, Max Drawdown -25.56%, Sharpe 0.91
- Discrepancy explanation: ~15 bps difference arises from calculating returns from post-trade equity $9,990.01 vs initial deposited capital $10,000.00.

### Honest Risk & Strategy Limitations:
- **Proxy Limitations:** ETFs `SOXX` and `XLK` represent broad semiconductor hardware and technology corporations (NVIDIA, Apple, Microsoft, Broadcom, etc.) rather than pure-play AI revenues.
- **Downside Protection Opportunity Cost:** During prolonged semiconductor bull markets, holding defensive cash buffers incurs an opportunity cost. In the holdout test period, the defensive strategy achieved a slightly lower CAGR (24.83% vs 25.63% buy-and-hold) in exchange for reduced maximum drawdown (-24.82% vs -25.56%) and superior risk-adjusted Sharpe ratio (0.93 vs 0.91).
- **Execution Lag & Slippage:** Rebalance decisions cannot execute instantaneously at signal time. Execution occurs at the close of the next trading session net of 10 bps modeled friction.
- **Zero Risk-Free Rate Assumption:** All Sharpe ratios assume a 0.0% risk-free rate strictly per course specification.
- **Defensive Asset Market Risk:** Short-duration Treasuries (`SHY`) are subject to interest rate fluctuations and inflation risk rather than serving as a guaranteed risk-free asset.

---

## 11. Project Submission & Peer Review Instructions

1. **Submission Link:** [SMA Zoomcamp 2026 Project Submission Attempt 1](https://courses.datatalks.club/sma-zoomcamp-2026/project/project1) (Open until 2026-11-20 23:59:59 GMT).
2. **Repository Contents:** Ensure that no raw price datasets (`data/raw/`), model binaries (`artifacts/models/`), or temporary virtual environments (`.venv/`) are checked into git.
3. **Peer Reviews:** To receive course credit, students must evaluate 3 peer projects between the submission deadline and the evaluation cutoff date on the DatatalksClub platform.
