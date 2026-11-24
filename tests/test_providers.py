"""Validate SDK request contracts without calling external services."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from clinical_trials.config import Settings
from clinical_trials.models import Criterion, Summary
from clinical_trials.providers import GeminiProvider


class GeminiContractTests(unittest.TestCase):
    def test_missing_credentials_fail_before_requests(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                GeminiProvider(Settings(model="generation", embedding_model="embedding"))

    def test_structured_assessment_and_embedding_tasks(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client") as client_factory:
                client = client_factory.return_value
                client.models.generate_content.return_value = SimpleNamespace(
                    text=json.dumps(
                        {
                            "eligibility": "Yes",
                            "evidence": ["Explicit diagnosis."],
                            "explanation": "The diagnosis is documented.",
                        }
                    )
                )
                client.models.embed_content.return_value = SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[0.1, 0.9])]
                )
                provider = GeminiProvider(Settings(model="generation", embedding_model="embedding"))
                assessment = provider.assess(
                    Summary(patient_id="one", text="Explicit diagnosis.", summary="Diagnosis."),
                    Criterion(trial_name="trial", name="criterion", text="Diagnosis"),
                    [],
                )
                self.assertEqual(assessment.eligibility, "Yes")
                request = client.models.generate_content.call_args.kwargs
                self.assertEqual(request["model"], "generation")
                self.assertEqual(request["config"].response_mime_type, "application/json")
                self.assertIn("Explicit diagnosis.", request["contents"])
                self.assertEqual(provider.embed(["Diagnosis"], "RETRIEVAL_QUERY"), [[0.1, 0.9]])
                embedding_request = client.models.embed_content.call_args.kwargs
                self.assertEqual(embedding_request["config"].task_type, "RETRIEVAL_QUERY")
                provider.close()
                client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
