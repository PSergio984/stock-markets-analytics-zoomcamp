"""Integration unit tests for Streamlit application using AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent.parent / "app.py"


def test_app_renders_default_compact_mode():
    """Assert app starts up in Compact mode without throwing unhandled exceptions."""
    at = AppTest.from_file(str(APP_PATH)).run()
    assert not at.exception, f"App raised an exception: {at.exception}"

    # Verify mode selector exists and defaults to Compact
    radios = at.radio
    assert len(radios) > 0
    assert radios[0].value == "Compact"


def test_app_switch_to_detailed_mode():
    """Assert switching to Detailed mode updates view without exception or changing identity."""
    at = AppTest.from_file(str(APP_PATH)).run()
    assert not at.exception

    # Select Detailed mode
    at.radio[0].set_value("Detailed").run()
    assert not at.exception
    assert at.radio[0].value == "Detailed"
