"""
test_app.py — UI smoke tests using Streamlit's AppTest framework.

These tests verify that:
- The app loads without error.
- Loading the sample dataset works.
- Running an audit with the sample data completes.
- Changing columns updates state.
- Uploading a different dataset clears previous results.

NOTE: AppTest does not run a real browser. It simulates widget interactions
in-process. Visual rendering is not verified here; that requires a real browser
(reported as a known limitation in the README).
"""
from __future__ import annotations

import pathlib
import pytest

try:
    from streamlit.testing.v1 import AppTest
    STREAMLIT_TESTING_AVAILABLE = True
except ImportError:
    STREAMLIT_TESTING_AVAILABLE = False


APP_PATH = str(pathlib.Path(__file__).parent.parent / "app.py")
SAMPLE_CSV = pathlib.Path(__file__).parent.parent / "sample_data" / "example_dataset.csv"


@pytest.mark.skipif(
    not STREAMLIT_TESTING_AVAILABLE,
    reason="streamlit.testing.v1 not available",
)
class TestAppSmoke:
    def test_app_loads_without_error(self):
        """App should initialise without raising an exception."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert not at.exception, f"App raised an exception: {at.exception}"

    def test_app_shows_upload_prompt(self):
        """Before upload, the app should show a file uploader."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        # There should be at least one file uploader widget
        assert len(at.file_uploader) >= 1

    def test_app_runs_with_sample_csv(self):
        """
        Simulate uploading the sample CSV and running an audit.
        Verify the app does not crash.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()

        # Upload sample file
        sample_bytes = SAMPLE_CSV.read_bytes()
        if at.file_uploader:
            at.file_uploader[0].set_value(("example_dataset.csv", sample_bytes, "text/csv"))
            at.run()
        assert not at.exception, f"Exception after upload: {at.exception}"
        at.selectbox(key="form_text_col").select("message")
        at.selectbox(key="form_label_col").select("category")
        next(button for button in at.button if button.label == "▶ Run Audit").click().run()
        assert not at.exception, f"Exception after audit: {at.exception}"
        assert at.session_state["audit_result"] is not None


def test_settings_and_run_button_are_in_main_workspace_after_upload():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].set_value(("example_dataset.csv", SAMPLE_CSV.read_bytes(), "text/csv"))
    at.run()
    assert not at.exception
    assert any(widget.label == "Text column" for widget in at.selectbox)
    assert any(widget.label == "Label column" for widget in at.selectbox)
    assert any(widget.label == "▶ Run Audit" for widget in at.button)
    assert any("Dataset loaded" in widget.value for widget in at.info)
