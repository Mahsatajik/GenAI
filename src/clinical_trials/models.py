"""Validated records retain patient and trial identity throughout the workflow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Note(Record):
    patient_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Summary(Record):
    patient_id: str
    text: str
    summary: str = ""
    error: str | None = None


class Criterion(Record):
    trial_name: str = Field(min_length=1)
    name: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: Literal["inclusion", "exclusion", "unspecified"] = "unspecified"
    lexicons: list[str] = Field(default_factory=list)
    error: str | None = None


class CriterionDetails(Record):
    text: str = Field(min_length=1)
    lexicons: list[str]


class Assessment(Record):
    eligibility: Literal["Yes", "No", "Unknown"]
    evidence: list[str] = Field(default_factory=list)
    explanation: str


class EligibilityResult(Assessment):
    patient_id: str
    trial_name: str
    criteria_name: str
    criterion_type: Literal["inclusion", "exclusion", "unspecified"]
    error: str | None = None


class Chunk(Record):
    chunk_id: str
    patient_id: str
    document: str
