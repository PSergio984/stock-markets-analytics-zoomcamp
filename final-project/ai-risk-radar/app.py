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
from ai_risk_radar.provenance import HISTORICAL_RELEASE_1_0_0
from ai_risk_radar.views import compact_view, detailed_view, format_percentage_points


def are_release_identities_compatible(primary_data: dict, repro_data: dict) -> bool:
    """Verify that primary report and reproduction report share the same complete release/experiment identity.

    Strict rules:
    1. Both must have non-empty identity structures.
    2. Missing or empty hashes are explicitly rejected.
    3. If release_hash is present in both and non-empty, they must match exactly.
    4. If release_hash is absent from either:
       - Both model_hash must match and be non-empty.
       - Any policy parameters (e.g. selected_threshold, weights) present in both must match exactly.
       - Any data/cutoff/partition markers present in both must match exactly.
       - For historical release 1.0.0, validated historical mapping must match HISTORICAL_RELEASE_1_0_0 exact parameters.
    5. Same model_hash with differing policy, data, or threshold is strictly rejected.
    """
    prim_id = primary_data.get("identity") or {}
    repro_id = repro_data.get("identity") or {}

    prim_rel_hash = prim_id.get("release_hash")
    repro_rel_hash = repro_id.get("release_hash")

    prim_model_hash = prim_id.get("model_hash")
    repro_model_hash = repro_id.get("model_hash")

    # 1. Require non-empty model identity
    if not prim_model_hash or not repro_model_hash:
        return False
    if not isinstance(prim_model_hash, str) or not isinstance(repro_model_hash, str):
        return False
    if not prim_model_hash.strip() or not repro_model_hash.strip():
        return False

    # Model hashes must match
    if prim_model_hash != repro_model_hash:
        return False

    # 2. Check compound release_hash if both have it
    if prim_rel_hash and repro_rel_hash:
        if not isinstance(prim_rel_hash, str) or not isinstance(repro_rel_hash, str):
            return False
        if not prim_rel_hash.strip() or not repro_rel_hash.strip():
            return False
        return prim_rel_hash == repro_rel_hash

    # 3. If release_hash is missing from either, verify full policy and data consistency
    prim_th = (
        primary_data.get("allocation", {}).get("selected_threshold")
        or prim_id.get("selected_threshold")
    )
    repro_th = (
        repro_data.get("allocation", {}).get("selected_threshold")
        or repro_id.get("selected_threshold")
    )
    if prim_th is not None and repro_th is not None:
        try:
            if abs(float(prim_th) - float(repro_th)) > 1e-6:
                return False  # Same model but different policy threshold!
        except (ValueError, TypeError):
            return False

    prim_cutoff = (
        primary_data.get("data_status", {}).get("cutoff_session")
        or prim_id.get("cutoff_session")
    )
    repro_cutoff = (
        repro_data.get("data_status", {}).get("cutoff_session")
        or repro_id.get("cutoff_session")
    )
    if prim_cutoff and repro_cutoff and prim_cutoff != repro_cutoff:
        return False  # Same model but different data cutoff!

    # Check against validated historical release identity mapping
    if repro_model_hash == HISTORICAL_RELEASE_1_0_0.model_hash:
        return not (
            prim_th is not None
            and abs(float(prim_th) - HISTORICAL_RELEASE_1_0_0.policy_parameters["selected_threshold"]) > 1e-6
        )

    return False


def load_report_data() -> tuple[Report | None, str | None]:
    """Load latest available report artifact, enriching latest inference with test metrics if needed."""
    artifacts_dir = Path("artifacts")
    latest_path = artifacts_dir / "latest_report.json"
    final_test_path = artifacts_dir / "final_test_report.json"
    repro_test_path = artifacts_dir / "reproduction_test_report.json"
    research_path = artifacts_dir / "research_report.json"

    primary_path = None
    if latest_path.exists():
        primary_path = latest_path
    elif repro_test_path.exists():
        primary_path = repro_test_path
    elif final_test_path.exists():
        primary_path = final_test_path
    elif research_path.exists():
        primary_path = research_path

    if not primary_path:
        return None, None

    try:
        with open(primary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check reproduction report and ONLY merge if release identities are strictly compatible
        if repro_test_path.exists():
            try:
                with open(repro_test_path, "r", encoding="utf-8") as rf:
                    repro_data = json.load(rf)
                if are_release_identities_compatible(data, repro_data):
                    data.setdefault(
                        "reproduction_status",
                        repro_data.get("reproduction_status") or repro_data.get("identity", {}).get("reproduction_status"),
                    )
                    repro_perf = repro_data.get("performance", {})
                    for k in (
                        "metric_convention",
                        "net_initial_capital_total_return",
                        "legacy_post_funding_total_return",
                        "funding_cost_impact_bps",
                        "verification_scope",
                        "verified_metrics",
                    ):
                        val = repro_perf.get(k) or repro_data.get(k) or repro_data.get("identity", {}).get(k)
                        if val is not None:
                            if "performance" in data and k not in data["performance"]:
                                data["performance"][k] = val
                            if k not in data:
                                data[k] = val
                    data["reproduction_verified"] = True
            except (OSError, json.JSONDecodeError):
                pass

        # If primary is latest inference without performance metrics, merge from test report ONLY if compatible
        candidates = [repro_test_path, final_test_path]
        for src_path in candidates:
            if src_path.exists():
                try:
                    with open(src_path, "r", encoding="utf-8") as f:
                        test_data = json.load(f)
                    if are_release_identities_compatible(data, test_data):
                        data["performance"] = test_data.get("performance", data.get("performance"))
                        data["comparisons"] = test_data.get("comparisons", data.get("comparisons"))
                        data["diagnostics"] = test_data.get("diagnostics", data.get("diagnostics"))
                        break
                except (OSError, json.JSONDecodeError):
                    pass

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

    # Performance Snapshot & Holdout Provenance
    metrics = c_view.get("selected_metrics")
    repro_status = c_view.get("reproduction_status")
    net_ret = c_view.get("net_initial_capital_total_return")
    legacy_ret = c_view.get("legacy_post_funding_total_return")
    funding_bps = c_view.get("funding_cost_impact_bps")

    if repro_status == "verified_match":
        st.success(
            "✅ **Holdout Reproduction: Verified Match** — Bit-exact trajectory equivalence verified on frozen test holdout."
        )

    if metrics:
        partition_name = c_view.get("evaluation_partition", "Test").title()
        st.subheader(f"📈 Performance Snapshot ({partition_name} Partition)")

        if net_ret is not None or legacy_ret is not None:
            r_cols = st.columns(3)
            if net_ret is not None:
                r_cols[0].metric(
                    label="Net Initial-Capital Return",
                    value=f"{net_ret:.1%}",
                    help="Total return based on $10,000 initial capital.",
                )
            if legacy_ret is not None:
                r_cols[1].metric(
                    label="Original Post-Funding Return",
                    value=f"{legacy_ret:.1%}",
                    help="Total return based on post-funding capital after session 0 setup cost ($9,990.01; legacy convention).",
                )
            if funding_bps is not None:
                r_cols[2].metric(
                    label="Initial Setup Cost Impact",
                    value=f"{funding_bps:.1f} bps",
                    help="Transaction cost difference from session 0 initial portfolio funding.",
                )

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
        repro_badge = f" | Reproduction: {repro_status}" if repro_status else ""
        st.caption(
            f"Artifact: `{filename}` | Model Hash: `{model_h[:8]}...` | Release Hash: `{release_h[:8]}...` | "
            f"Pipeline Mode: `{c_view.get('identity', {}).get('pipeline_mode', 'N/A')}`{repro_badge}"
        )


if __name__ == "__main__":
    main()
