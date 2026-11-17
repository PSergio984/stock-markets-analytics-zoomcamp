# AI Investment Risk Radar

An end-to-end, reproducible quantitative risk forecasting system and defensive allocation dashboard built for the Stock Market Analytics Zoomcamp final project.

## Problem Statement

Investments in artificial intelligence and semiconductor hardware (represented by the **SOXX** semiconductor ETF and **XLK** technology ETF) have experienced substantial volatility and acute drawdowns alongside immense upside. For beginner and individual investors, navigating these sector downturns without panicking or suffering severe capital impairment is a critical challenge.

The **AI Investment Risk Radar** solves this problem by:
1. **Defining an explicit downside loss event**: Forecasting the probability that an AI-sensitive reference basket (50% SOXX / 50% XLK) ends the next 20 trading days with a total return of `<= -5.0%`. This measures *end-of-horizon loss probability*, explicitly distinguished from intra-horizon maximum drawdown.
2. **Delayed Defensive Allocation Simulation**: When risk forecasts exceed an empirically selected threshold, the strategy shifts portfolio exposure from a growth allocation (35% SOXX, 35% XLK, 20% SPY, 10% SHY) to a defensive allocation (10% SOXX, 15% XLK, 35% SPY, 40% SHY).
3. **Execution Realism**: Rebalance signals occur at the final completed trading session of each week and execute at the *next* trading session's close. Positions drift between rebalances, and realistic round-trip transaction costs (10 bps baseline, plus 5 bps and 25 bps sensitivity) are deducted directly from portfolio equity.
4. **Honest Reporting & Evaluation**: Incorporates strict chronological splits (4+ years training, 12 months probability calibration, 24 months validation, and 24 months untouched holdout test) with 20-session label purging to prevent temporal leakage. A two-mode interactive dashboard (Compact and Detailed) presents identical computed outcomes without fabricated metrics.

## Fixed Asset Universe

| Ticker | Asset Description | Role in Strategy |
|---|---|---|
| **SOXX** | iShares Semiconductor ETF | AI hardware & semiconductor sector proxy |
| **XLK** | Technology Select Sector SPDR Fund | Broad technology & software sector proxy |
| **SPY** | SPDR S&P 500 ETF Trust | Broad US equity market diversification |
| **SHY** | iShares 1-3 Year Treasury Bond ETF | Short-duration risk-free capital preservation |

Reference AI basket: `50% SOXX + 50% XLK` (buy-and-hold through each 20-session forecast horizon).

## Environment Setup

The project uses Python 3.12+ and isolated pinned dependencies managed via `uv` or `pip`.

### Quickstart with `uv` (Recommended)

```bash
cd final-project/ai-risk-radar
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip sync requirements.txt
uv pip install -e .
```

### Quickstart with standard `python -m venv`

```bash
cd final-project/ai-risk-radar
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Running Tests

Run deterministic unit and financial correctness tests:

```bash
python -m pytest tests/ -v
python -m ruff check .
```

## Data Acquisition and Snapshots

To fetch fresh price data from yfinance aligned with the New York Stock Exchange (`XNYS`) calendar:

```bash
python -m ai_risk_radar fetch --config config.json
```

Or run via the pipeline commands detailed in subsequent sections.
Raw datasets and cached binaries are strictly excluded from git tracking via `.gitignore`.
