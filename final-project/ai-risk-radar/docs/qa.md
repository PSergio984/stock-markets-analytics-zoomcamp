# AI Investment Risk Radar: QA & Visual Verification Report

This document records the visual and automated QA verification for Task 6 (Two-Mode Dashboard) of the AI Investment Risk Radar application.

---

## 1. Automated App Testing (`streamlit.testing.v1.AppTest`)

The test suite at `tests/test_app.py` exercises the Streamlit component tree in memory without opening a physical browser:
- `test_app_renders_default_compact_mode`: Verified that `app.py` boots in `Compact` mode with 0 unhandled exceptions and displays all KPI metrics, target allocations, and risk definitions.
- `test_app_switch_to_detailed_mode`: Verified that toggling the view mode radio widget to `Detailed` renders the full quantitative cockpit (comparisons matrix, reliability diagnostics, causal feature list, limitations) without modifying any identity, risk level, or performance values.

Execution proof:
```
tests/test_app.py::test_app_renders_default_compact_mode PASSED [ 50%]
tests/test_app.py::test_app_switch_to_detailed_mode PASSED      [100%]
2 passed in 3.76s
```

---

## 2. Viewport & Layout Emulation

Visual inspection was performed using headless Chromium across desktop and mobile viewports against a locally running Streamlit instance on port 8502:

| Viewport Mode | Window Dimensions | Artifact | Verification Notes |
| :--- | :--- | :--- | :--- |
| **Desktop (Compact)** | 1280 x 900 px | `docs/screenshots/compact-desktop.png` | Zero horizontal scroll. Radio selector, risk badge, target allocation columns, and KPI metrics aligned cleanly. |
| **Mobile (Compact)** | 375 x 812 px | `docs/screenshots/compact-mobile.png` | Responsive multi-column collapse. Allocation metrics stack vertically with readable typography and zero content clipping. |

---

## 3. Visual Invariant Checklist

- [x] **Default Mode:** Application defaults to `Compact` view on initial load.
- [x] **Shared Identity:** Switching to `Detailed` view preserves exact same as-of session, risk probability (`45.4%`), risk level (`High`), strategy mode (`Growth`), and performance metrics.
- [x] **Loss Definition Transparency:** Downside risk event is explicitly defined as `Probability of reference basket (50% SOXX / 50% XLK) total return <= -5.0% over next 20 trading sessions`.
- [x] **Units Precision:** Total return differences between strategies are formatted in percentage points (`pp`), avoiding deceptive relative percentage representations.
- [x] **Honest Disclaimers:** Visible callout documents proxy ETF limitations, execution delay at next-session close with 10 bps slippage, and zero risk-free Sharpe ratio convention.
