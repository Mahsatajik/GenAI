"""Exercise upload parsing and the UI callback using synthetic files."""

from pathlib import Path
import tempfile
import unittest

from clinical_trials.ui import build_app, submission_notes
from clinical_trials.config import Settings


class UploadTests(unittest.TestCase):
    def test_text_upload_and_manual_note(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "note.txt"
            path.write_text("A fictional note.", encoding="utf-8")
            notes = submission_notes("Another fictional note.", str(path))
            self.assertEqual([note.patient_id for note in notes], ["manual-note", "uploaded-note"])

    def test_empty_upload(self):
        with self.assertRaises(ValueError):
            submission_notes("", None)

    def test_build_has_screening_api_and_multiple_criteria_upload(self):
        app = build_app(Settings(), demo_mode=True)
        config = app.get_config_file()
        uploads = [
            component
            for component in config["components"]
            if component.get("props", {}).get("label") == "Trial criteria"
        ]
        self.assertEqual(uploads[0]["props"]["file_count"], "multiple")
        self.assertTrue(any(fn.api_name == "screen" for fn in app.fns.values()))


if __name__ == "__main__":
    unittest.main()
