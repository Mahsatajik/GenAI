"""Evidence checks and reverse-retrieval diagnostics, distinct from clinical accuracy."""

import numpy as np
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

from .models import Assessment, EligibilityResult, Summary
from .retrieval import ChunkIndex


COLUMNS = [
    "Trial_Name",
    "Criteria_Name",
    "Criterion_Type",
    "Patient_id",
    "Eligibility",
    "Evidence",
    "MatchedChunk",
    "Patient_Summary",
    "EvidenceInOriginalNote",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "fuzz_MatchScore",
    "CosineSimilarity",
    "Error",
]


def check_evidence(assessment: Assessment, original_note: str) -> Assessment:
    evidence = [item.strip() for item in assessment.evidence if item.strip()]
    invalid = any(item not in original_note for item in evidence)
    if invalid or (assessment.eligibility != "Unknown" and not evidence):
        return Assessment(
            eligibility="Unknown",
            evidence=[],
            explanation="The response lacked an exact supporting quotation from this patient's note.",
        )
    return assessment.model_copy(update={"evidence": evidence})


def evaluate(
    results: list[EligibilityResult],
    summaries: list[Summary],
    index: ChunkIndex,
) -> list[dict]:
    summary_map = {item.patient_id: item for item in summaries}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for result in results:
        patient = summary_map[result.patient_id]
        for evidence in result.evidence or [None]:
            row = dict.fromkeys(COLUMNS)
            row.update(
                {
                    "Trial_Name": result.trial_name,
                    "Criteria_Name": result.criteria_name,
                    "Criterion_Type": result.criterion_type,
                    "Patient_id": result.patient_id,
                    "Eligibility": result.eligibility,
                    "Evidence": evidence,
                    "Patient_Summary": patient.summary,
                    "EvidenceInOriginalNote": evidence in patient.text if evidence else None,
                    "Error": result.error,
                }
            )
            if evidence:
                try:
                    matched = index.query(result.patient_id, evidence, limit=1)
                    if matched:
                        chunk = matched[0].document
                        scores = scorer.score(evidence, chunk)
                        row.update(
                            {
                                "MatchedChunk": chunk,
                                "ROUGE-1": round(scores["rouge1"].fmeasure * 100, 2),
                                "ROUGE-2": round(scores["rouge2"].fmeasure * 100, 2),
                                "ROUGE-L": round(scores["rougeL"].fmeasure * 100, 2),
                                "fuzz_MatchScore": round(fuzz.partial_ratio(evidence, chunk), 2),
                            }
                        )
                        query = np.asarray(index.embed([evidence], "RETRIEVAL_QUERY")[0])
                        document = np.asarray(index.embed([chunk], "RETRIEVAL_DOCUMENT")[0])
                        cosine = np.dot(query, document) / (
                            np.linalg.norm(query) * np.linalg.norm(document)
                        )
                        row["CosineSimilarity"] = round(float(cosine) * 100, 2)
                except Exception as error:
                    row["Error"] = f"Evidence diagnostics failed ({type(error).__name__})."
            rows.append(row)
    return rows
