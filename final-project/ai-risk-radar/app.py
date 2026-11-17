"""Streamlit dashboard for AI Investment Risk Radar.

Supports two view modes:
- Compact View (default): scannable risk gauge, downside probability, target weights, trade action.
- Detailed View: quantitative cockpit with benchmark comparisons, reliability diagnostics, failure regimes.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_risk_radar.contracts import Report
from ai_risk_radar.views import compact_view, detailed_view, format_percentage_points


def load_report_data() -> tuple[Report | None, str | None]:
    """Load latest available report artifact, enriching latest inference with test metrics if needed."""
    artifacts_dir = Path("artifacts")
    latest_path = artifacts_dir / "latest_report.json"
    final_test_path = artifacts_dir / "final_test_report.json"
    research_path = artifacts_dir / "research_report.json"

    primary_path = None
    if latest_path.exists():
        primary_path = latest_path
    elif final_test_path.exists():
        primary_path = final_test_path
    elif research_path.exists():
        primary_path = research_path

    if not primary_path:
        return None, None

    try:
        with open(primary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If primary is latest inference without performance metrics, merge from final_test_report
        if (
            data.get("performance", {}).get("selected_strategy") is None
            and final_test_path.exists()
        ):
            with open(final_test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
                data["performance"] = test_data.get("performance", data.get("performance"))
                data["comparisons"] = test_data.get("comparisons", data.get("comparisons"))
                data["diagnostics"] = test_data.get("diagnostics", data.get("diagnostics"))

        return Report(payload=data), str(primary_path.name)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Error loading report from {primary_path}: {exc}")
        return None, None


def main() -> None:
    st.set_page_config(
        page_title="AI Investment Risk Radar",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Top header bar
    st.title("📡 AI Investment Risk Radar")
    st.caption(
        "Causal ETF downside rotation radar for AI concentration (SOXX / XLK / SPY / SHY)"
    )

    # Mode Selector
    default_idx = 1 if st.query_params.get("view", "").lower() == "detailed" else 0
    mode = st.radio(
        "View Mode",
        options=["Compact", "Detailed"],
        index=default_idx,
        horizontal=True,
        help="Compact is designed for quick daily briefings; Detailed provides full quantitative diagnostics.",
    )

    report, filename = load_report_data()

    if report is None:
        st.warning(
            "⚠️ No report artifacts found in `artifacts/`. "
            "Please run `python -m ai_risk_radar latest` or `python -m ai_risk_radar final-test`."
        )
        return

    c_view = compact_view(report)
    d_view = detailed_view(report)

    # Warning banners
    if c_view["is_stale"]:
        st.warning(
            f"⚠️ **Market Data Stale:** Snapshot as of `{c_view['as_of_session']}` "
            f"is {c_view['staleness_sessions']} session(s) behind market close."
        )

    if c_view["probability_status"] == "unavailable":
        st.error(
            f"⚠️ **Risk Probability Unavailable:** {c_view['fallback_reason']}. "
            "Strategy has safely fallen back to Fixed Baseline allocation."
        )

    # Common Compact View Content
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric(
            label="Risk Level",
            value=c_view["risk_level"],
            help="High (>=35%), Medium (20-35%), Low (<20%), or Unavailable.",
        )

    with col2:
        st.metric(
            label="Downside Risk Probability",
            value=c_view["probability_display"],
            help="Estimated probability of 50/50 SOXX/XLK basket losing >= 5% over next 20 trading sessions.",
        )

    with col3:
        st.markdown(f"**As-of Session:** `{c_view['as_of_session']}`")
        st.markdown(f"**Event Definition:** {c_view['event_definition']}")
        if c_view["cutoff_session"]:
            st.caption(f"Completed session cutoff: `{c_view['cutoff_session']}`")

    # Allocation & Action Section
    st.subheader("🎯 Target Portfolio Allocation")
    alloc = c_view["allocation"]
    weights = alloc.get("weights", {})
    strategy_mode = alloc.get("strategy_mode", "Unknown")
    trade_desc = alloc.get("trade_description", "No description available.")

    st.info(f"**Current Stance:** `{strategy_mode}` — {trade_desc}")

    # Display allocation weights in columns
    w_cols = st.columns(4)
    w_cols[0].metric("SOXX (Semis)", f"{weights.get('SOXX', 0.0):.1%}")
    w_cols[1].metric("XLK (Tech)", f"{weights.get('XLK', 0.0):.1%}")
    w_cols[2].metric("SPY (S&P 500)", f"{weights.get('SPY', 0.0):.1%}")
    w_cols[3].metric("SHY (1-3y Treasuries)", f"{weights.get('SHY', 0.0):.1%}")

    # Performance Snapshot
    metrics = c_view.get("selected_metrics")
    if metrics:
        partition_name = c_view.get("evaluation_partition", "Test").title()
        st.subheader(f"📈 Performance Snapshot ({partition_name} Partition)")
        p_cols = st.columns(5)
        p_cols[0].metric("CAGR", f"{metrics.get('cagr', 0.0):.1%}")
        p_cols[1].metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.1%}")
        p_cols[2].metric("Sharpe (rf=0)", f"{metrics.get('sharpe', 0.0):.2f}")
        p_cols[3].metric("Turnover", f"{metrics.get('turnover', 0.0):.2f}x")
        p_cols[4].metric("Defensive Occupancy", f"{metrics.get('defensive_occupancy', 0.0):.1%}")

    # Additional sections for Detailed View
    if mode == "Detailed":
        st.markdown("---")
        st.subheader("📊 Benchmark Strategy Comparison Matrix")

        comparisons = d_view.get("comparisons", {})
        if comparisons:
            rows = []
            strat_return = metrics.get("total_return") if metrics else None

            # Add Selected Strategy row
            if metrics:
                rows.append({
                    "Strategy": f"Selected Strategy (Threshold {alloc.get('selected_threshold', 0.50):.2f})",
                    "Total Return": f"{metrics.get('total_return', 0.0):.1%}",
                    "Return Diff (pp)": "Base",
                    "CAGR": f"{metrics.get('cagr', 0.0):.1%}",
                    "Max Drawdown": f"{metrics.get('max_drawdown', 0.0):.1%}",
                    "Sharpe (rf=0)": f"{metrics.get('sharpe', 0.0):.2f}",
                    "Turnover": f"{metrics.get('turnover', 0.0):.2f}x",
                    "Defensive Time": f"{metrics.get('defensive_occupancy', 0.0):.1%}",
                })

            for name, c_data in comparisons.items():
                label = name.replace("_", " ").title()
                diff_str = format_percentage_points(c_data.get("total_return"), strat_return)
                rows.append({
                    "Strategy": label,
                    "Total Return": f"{c_data.get('total_return', 0.0):.1%}",
                    "Return Diff (pp)": diff_str,
                    "CAGR": f"{c_data.get('cagr', 0.0):.1%}",
                    "Max Drawdown": f"{c_data.get('max_drawdown', 0.0):.1%}",
                    "Sharpe (rf=0)": f"{c_data.get('sharpe', 0.0):.2f}",
                    "Turnover": f"{c_data.get('turnover', 0.0):.2f}x",
                    "Defensive Time": f"{c_data.get('defensive_occupancy', 0.0):.1%}",
                })

            df_comp = pd.DataFrame(rows)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Model & Calibration Diagnostics
        st.subheader("🔬 Model & Calibration Diagnostics")
        diag = d_view.get("diagnostics", {})
        test_diag = diag.get("test_classification", {})

        d_cols = st.columns(3)
        brier = test_diag.get("brier_score")
        non_ov_brier = test_diag.get("non_overlapping", {}).get("brier_score")
        prev = test_diag.get("prevalence")

        d_cols[0].metric("Test Brier Score", f"{brier:.3f}" if brier is not None else "N/A")
        d_cols[1].metric(
            "Non-Overlapping Brier",
            f"{non_ov_brier:.3f}" if non_ov_brier is not None else "N/A",
        )
        d_cols[2].metric(
            "Downside Event Prevalence",
            f"{prev:.1%}" if prev is not None else "N/A",
        )

        # Reliability Bins
        bins = test_diag.get("reliability_bins", [])
        if bins:
            st.markdown("**Test Calibration Reliability Bins:**")
            bin_rows = []
            for b in bins:
                bin_rows.append({
                    "Probability Range": b.get("bin"),
                    "Observations": b.get("count"),
                    "Mean Predicted": (
                        f"{b['pred_mean']:.1%}" if b.get("pred_mean") is not None else "-"
                    ),
                    "Mean Observed": (
                        f"{b['obs_mean']:.1%}" if b.get("obs_mean") is not None else "-"
                    ),
                })
            st.dataframe(pd.DataFrame(bin_rows), use_container_width=True, hide_index=True)

        # Causal Feature Drivers
        st.subheader("💡 Causal Feature Drivers")
        for feat in c_view.get("feature_explanation", []):
            st.markdown(f"- {feat}")

        # Failure Regimes & Limitations
        st.subheader("⚠️ Documented Limitations & Failure Regimes")
        for lim in d_view.get("limitations", []):
            st.markdown(f"- {lim}")

        # Provenance
        model_h = c_view.get("identity", {}).get("model_hash") or c_view.get("identity", {}).get("release_hash", "N/A")
        release_h = c_view.get("identity", {}).get("release_hash", "N/A")
        st.caption(
            f"Artifact: `{filename}` | Model Hash: `{model_h[:8]}...` | Release Hash: `{release_h[:8]}...` | "
            f"Pipeline Mode: `{c_view.get('identity', {}).get('pipeline_mode', 'N/A')}`"
        )


if __name__ == "__main__":
    main()
