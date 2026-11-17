"""View formatters and presentation transformers for Compact and Detailed dashboard modes."""

from ai_risk_radar.contracts import Report


def format_percentage_points(val_a: float | None, val_b: float | None) -> str:
    """Format the difference between two rates in percentage points (pp)."""
    if val_a is None or val_b is None:
        return "N/A"
    diff = (val_a - val_b) * 100.0
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f} pp"


def compact_view(report: Report) -> dict[str, object]:
    """Transform report artifact into structured dictionary for Compact view."""
    payload = report.payload
    forecast = payload.get("forecast", {})
    data_status = payload.get("data_status", {})
    allocation = payload.get("allocation", {})
    performance = payload.get("performance", {})
    identity = payload.get("identity", {})

    prob = forecast.get("probability")
    prob_status = forecast.get("probability_status", "uncalibrated")
    fallback_reason = forecast.get("fallback_reason")

    if prob is not None and prob_status != "unavailable":
        prob_display = f"{float(prob):.1%}"
        risk_level = forecast.get("risk_level", "Medium")
    else:
        prob_display = "Unavailable"
        risk_level = "Unavailable"

    staleness = int(data_status.get("staleness_sessions", 0))
    is_stale = staleness > 0

    return {
        "as_of_session": data_status.get("as_of_session", "Unknown"),
        "cutoff_session": data_status.get("cutoff_session"),
        "probability": prob,
        "probability_display": prob_display,
        "probability_status": prob_status,
        "fallback_reason": fallback_reason,
        "risk_level": risk_level,
        "event_definition": forecast.get(
            "loss_definition",
            "Probability of reference basket (50% SOXX / 50% XLK) total return <= -5.0% over next 20 trading sessions",
        ),
        "feature_explanation": forecast.get("feature_explanation", []),
        "is_stale": is_stale,
        "staleness_sessions": staleness,
        "allocation": allocation,
        "selected_metrics": performance.get("selected_strategy"),
        "evaluation_partition": performance.get("evaluation_partition", "validation"),
        "identity": identity,
    }


def detailed_view(report: Report) -> dict[str, object]:
    """Transform report artifact into structured dictionary for Detailed view."""
    base = compact_view(report)
    payload = report.payload

    comparisons = payload.get("comparisons", {})
    diagnostics = payload.get("diagnostics", {})
    limitations = payload.get("limitations", [])

    base["comparisons"] = comparisons
    base["diagnostics"] = diagnostics
    base["limitations"] = limitations

    return base
