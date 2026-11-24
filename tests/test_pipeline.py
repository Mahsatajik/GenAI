"""Regression checks for patient identity, evidence provenance, and run isolation."""

import csv
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from clinical_trials.config import Settings
from clinical_trials.data import read_criteria, read_notes
from clinical_trials.evaluation import check_evidence
from clinical_trials.models import Assessment, Criterion, Note
from clinical_trials.pipeline import run_pipeline
from clinical_trials.providers import DemoProvider
from clinical_trials.retrieval import ChunkIndex, split_text
from clinical_trials.storage import SummaryStore, filename


class FailedFirstSummary(DemoProvider):
    def summarize(self, text):
        if text == "fail":
            raise RuntimeError("Simulated failure")
        return super().summarize(text)


class UnsupportedEvidence(DemoProvider):
    def assess(self, patient, criterion, chunks):
        return Assessment(
            eligibility="Yes", evidence=["Invented evidence."], explanation="Simulated response."
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.settings = Settings(output_dir=self.root)
        self.criterion = Criterion(trial_name="trial-a", name="Inclusion_1", text="type 2 diabetes")
        self.notes = [
            Note(patient_id="one", text="The patient has type 2 diabetes."),
            Note(patient_id="two", text="The patient has a sprained ankle."),
        ]

    def tearDown(self):
        self.temp.cleanup()

    def test_pipeline_retains_alignment_after_summary_failure(self):
        notes = [Note(patient_id="failed", text="fail"), self.notes[0]]
        result = run_pipeline(notes, [self.criterion], self.settings, FailedFirstSummary())
        summaries = result.state["summaries"]
        self.assertEqual([item.patient_id for item in summaries], ["failed", "one"])
        self.assertEqual(summaries[1].summary, self.notes[0].text)
        decisions = result.state["eligibility_results"]
        self.assertEqual([item.eligibility for item in decisions], ["Unknown", "Yes"])
        self.assertTrue(result.has_errors)
        rows = {row["id"]: row for row in result.database_rows}
        self.assertEqual(rows["one"]["original_text"], self.notes[0].text)

    def test_cross_patient_retrieval_and_same_named_criteria(self):
        other = self.criterion.model_copy(update={"trial_name": "trial-b"})
        result = run_pipeline(self.notes, [self.criterion, other], self.settings, DemoProvider())
        decisions = result.state["eligibility_results"]
        self.assertEqual(len(decisions), 4)
        self.assertEqual(
            [item.eligibility for item in decisions], ["Yes", "Yes", "Unknown", "Unknown"]
        )
        self.assertEqual(len(list((result.output_dir / "evaluation").glob("*/*.csv"))), 2)
        with ZipFile(result.output_dir / "report.zip") as archive:
            self.assertIn("evaluation/all_results.csv", archive.namelist())
            self.assertIn("eligibility.json", archive.namelist())
            self.assertFalse(any(name.startswith("chromadb/") for name in archive.namelist()))
        with result.csv_path.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_retrieval_filters_patient_metadata(self):
        from clinical_trials.models import Summary

        index = ChunkIndex(DemoProvider(), self.root / "index", self.settings)
        index.add(
            [
                Summary(patient_id="one", text=self.notes[0].text, summary=self.notes[0].text),
                Summary(patient_id="two", text=self.notes[1].text, summary=self.notes[1].text),
            ]
        )
        chunks = index.query("two", "type 2 diabetes")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].patient_id, "two")
        self.assertNotIn("diabetes", chunks[0].document)
        self.assertEqual(index.query("absent", "diabetes"), [])

    def test_invented_evidence_becomes_unknown(self):
        result = run_pipeline(
            self.notes[:1], [self.criterion], self.settings, UnsupportedEvidence()
        )
        decision = result.state["eligibility_results"][0]
        self.assertEqual(decision.eligibility, "Unknown")
        self.assertEqual(decision.evidence, [])
        self.assertTrue(decision.error)

    def test_runs_are_independent(self):
        first = run_pipeline(self.notes[:1], [self.criterion], self.settings, DemoProvider())
        second = run_pipeline(self.notes[1:], [self.criterion], self.settings, DemoProvider())
        self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertEqual([row["id"] for row in second.database_rows], ["two"])
        store = SummaryStore(self.root / "clinical_data.db")
        self.assertEqual([row["id"] for row in store.rows(first.run_id)], ["one"])


class InputTests(unittest.TestCase):
    def test_duplicate_patient_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "notes.csv"
            path.write_text("patient_id,text\none,First note\none,Second note\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                read_notes(path)

    def test_legacy_csv_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "notes.csv"
            path.write_text("text\n\nA fictional clinical note\n", encoding="utf-8")
            self.assertEqual(read_notes(path)[0].patient_id, "patient-0001")
            trial = root / "trial.csv"
            trial.write_text("Criteria_name,Content\nExclusion_1,Condition X\n", encoding="utf-8")
            criteria = read_criteria([trial])
            self.assertEqual(criteria[0].kind, "exclusion")
            self.assertEqual(criteria[0].trial_name, "trial")

    def test_output_names_are_safe_and_collision_resistant(self):
        self.assertNotIn("/", filename("../../other"))
        self.assertNotEqual(filename("a/b"), filename("a b"))

    def test_invalid_chunk_overlap(self):
        with self.assertRaises(ValueError):
            split_text("a sentence", 4, 4)
        self.assertEqual(split_text("", 450, 50), [])
        chunks = split_text("Word " * 200, 100, 20)
        self.assertTrue(all(0 < len(chunk) <= 100 for chunk in chunks))

    def test_missing_evidence_cannot_support_yes_or_no(self):
        for decision in ("Yes", "No"):
            assessment = Assessment(eligibility=decision, evidence=[], explanation="No evidence.")
            self.assertEqual(check_evidence(assessment, "Original note").eligibility, "Unknown")


if __name__ == "__main__":
    unittest.main()
