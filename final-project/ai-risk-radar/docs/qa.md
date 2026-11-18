# AI Investment Risk Radar: QA & Visual Verification Report

This document records the visual, layout, and automated QA verification for Task 6 and Round 1 review findings of the AI Investment Risk Radar application.

---

## 1. Automated App Testing (`streamlit.testing.v1.AppTest`)

The test suite at `tests/test_app.py` exercises the Streamlit component tree in memory:
- `test_app_renders_default_compact_mode`: Verified that `app.py` boots in `Compact` mode with 0 unhandled exceptions and displays all KPI metrics, target allocations, and risk definitions.
- `test_app_switch_to_detailed_mode`: Verified that toggling the view mode radio widget to `Detailed` renders the full quantitative cockpit (comparisons matrix, reliability diagnostics, causal feature list, limitations) without modifying any identity, risk level, or performance values.

Execution proof:
```
tests/test_app.py::test_app_renders_default_compact_mode PASSED [ 50%]
tests/test_app.py::test_app_switch_to_detailed_mode PASSED      [100%]
2 passed in 3.76s
```

---

## 2. Viewport Measurements & Layout Emulation

Visual inspection was performed using headless Chromium via Chrome DevTools Protocol (CDP) WebSocket communication across desktop and mobile viewports against a locally running Streamlit instance on port 8503.

To eliminate timing race conditions where standard headless Chrome `--screenshot` captured premature loading states before React/WebSocket hydration (identified in Round 2 review), capture scripts polled `Runtime.evaluate` to confirm required DOM content (`Downside Risk Probability`, `Benchmark Strategy Comparison Matrix`) was visible prior to invoking `Page.captureScreenshot`:

| Viewport Mode | Window Dimensions | `clientWidth` | `scrollWidth` | Overflow Status | Artifact Screenshot | Verification Notes |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Desktop (Compact)** | 1280 x 900 px | 1264 px | 1264 px | None (PASS) | `docs/screenshots/compact-desktop.png` | Centered layout container (~704px content width within 1264px client viewport, 16px vertical scroll gutter). Full KPI cards, risk level ('High'), 45.4% probability, and target allocation cleanly rendered (84.8 KB). |
| **Mobile (Compact)** | 375 x 812 px | 375 px | 375 px | None (PASS) | `docs/screenshots/compact-mobile.png` | Emulated mobile viewport (`Emulation.setDeviceMetricsOverride`). Multi-column allocation and KPI metrics collapse vertically into single column with zero horizontal clipping (53.2 KB). |
| **Desktop (Detailed)** | 1280 x 1400 px | 1264 px | 1264 px | None (PASS) | `docs/screenshots/detailed-desktop.png` | Benchmark strategy matrix, Brier diagnostics, reliability bins, causal features, and limitations cleanly rendered (146.6 KB). |
| **Mobile (Detailed)** | 375 x 1400 px | 375 px | 375 px | Contained table scroll (PASS) | `docs/screenshots/detailed-mobile.png` | Benchmark table contained in responsive horizontal scroll wrapper; root page document element does not overflow (95.6 KB). |

---

## 3. Visual & Functional Invariant Checklist

- [x] **Default Mode:** Application defaults to `Compact` view on initial load without query parameters.
- [x] **Query Param Deep-Linking:** URL parameter `?view=detailed` correctly defaults to Detailed mode for direct bookmarks and headless audit tools.
- [x] **Shared Identity:** Switching to `Detailed` view preserves exact same as-of session, risk probability (`45.4%`), risk level (`High`), strategy mode (`Growth`), and performance metrics.
- [x] **Dynamic Subheader:** Performance snapshot subheader dynamically adapts to loaded partition: e.g., `📈 Performance Snapshot (Test Partition)` or `📈 Performance Snapshot (Validation Partition)`.
- [x] **Risk Badge Tooltip:** Risk level badge tooltip accurately reflects implemented thresholds: `High (>=35%), Medium (20-35%), Low (<20%), or Unavailable.`
- [x] **Loss Definition Transparency:** Downside risk event is explicitly defined as `Probability of reference basket (50% SOXX / 50% XLK) total return <= -5.0% over next 20 trading sessions`.
- [x] **Units Precision:** Total return differences between strategies are formatted in percentage points (`pp`), avoiding deceptive relative percentage representations.
- [x] **Honest Disclaimers:** Visible callout documents proxy ETF limitations, execution delay at next-session close with 10 bps slippage, and zero risk-free Sharpe ratio convention.
