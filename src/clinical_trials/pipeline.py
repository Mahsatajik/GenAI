"""One LangGraph workflow drives CLI, Python API, and Gradio execution."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Callable, TypedDict
from uuid import uuid4
from zipfile import ZipFile

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .data import validate_notes, write_csv, write_json
from .evaluation import COLUMNS, check_evidence, evaluate
from .models import Criterion, EligibilityResult, Note, Summary
from .providers import Provider
from .retrieval import ChunkIndex
from .storage import SummaryStore, filename


logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    notes: list[Note]
    summaries: list[Summary]
    criteria: list[Criterion]
    eligibility_results: list[EligibilityResult]
    evaluation_results: list[dict]


@dataclass
class RunResult:
    run_id: str
    output_dir: Path
    state: PipelineState
    chunks: list[dict]
    database_rows: list[dict]
    stages: list[str] = field(default_factory=list)

    @property
    def csv_path(self) -> Path:
        return self.output_dir / "evaluation" / "all_results.csv"

    @property
    def has_errors(self) -> bool:
        return (
            any(item.error for item in self.state["summaries"])
            or any(item.error for item in self.state["criteria"])
            or any(item.error for item in self.state["eligibility_results"])
            or any(row["Error"] for row in self.state["evaluation_results"])
        )


def build_workflow(provider: Provider, index: ChunkIndex, store: SummaryStore, run_id: str):
    def summarize(state):
        summaries = []
        for note in state["notes"]:
            try:
                text = provider.summarize(note.text).strip()
                if not text:
                    raise ValueError("Empty summary.")
                summary = Summary(patient_id=note.patient_id, text=note.text, summary=text)
            except Exception as error:
                summary = Summary(
                    patient_id=note.patient_id,
                    text=note.text,
                    error=f"Summarization failed ({type(error).__name__}).",
                )
            summaries.append(summary)
        return {"summaries": summaries}

    def persist(state):
        store.save(run_id, state["summaries"])
        return {}

    def prepare(state):
        criteria = []
        for criterion in state["criteria"]:
            try:
                details = provider.prepare_criterion(criterion)
                criterion = criterion.model_copy(
                    update={"text": details.text, "lexicons": details.lexicons}
                )
            except Exception as error:
                criterion = criterion.model_copy(
                    update={"error": f"Criteria preparation failed ({type(error).__name__})."}
                )
            criteria.append(criterion)
        return {"criteria": criteria}

    def embed(state):
        index.add(state["summaries"])
        return {}

    def assess(state):
        results = []
        for patient in state["summaries"]:
            for criterion in state["criteria"]:
                error_message = patient.error or criterion.error
                assessment = None
                if error_message is None:
                    try:
                        chunks = index.query(patient.patient_id, criterion.text)
                        if not chunks:
                            raise ValueError("No patient-specific chunks available.")
                        assessment = provider.assess(patient, criterion, chunks)
                        checked = check_evidence(assessment, patient.text)
                        if checked != assessment:
                            # Track rejected evidence separately from a valid Unknown decision.
                            if checked.eligibility != assessment.eligibility:
                                error_message = "The model response failed evidence validation."
                        assessment = checked
                    except Exception as error:
                        error_message = f"Criterion assessment failed ({type(error).__name__})."
                values = (
                    assessment.model_dump()
                    if assessment
                    else {
                        "eligibility": "Unknown",
                        "evidence": [],
                        "explanation": "This criterion could not be evaluated.",
                    }
                )
                results.append(
                    EligibilityResult(
                        **values,
                        patient_id=patient.patient_id,
                        trial_name=criterion.trial_name,
                        criteria_name=criterion.name,
                        criterion_type=criterion.kind,
                        error=error_message,
                    )
                )
        return {"eligibility_results": results}

    def diagnostics(state):
        return {
            "evaluation_results": evaluate(state["eligibility_results"], state["summaries"], index)
        }

    graph = StateGraph(PipelineState)
    nodes = [
        ("summarization", summarize),
        ("database", persist),
        ("criteria", prepare),
        ("chunking_embedding", embed),
        ("eligibility", assess),
        ("evaluation", diagnostics),
    ]
    previous = START
    for name, function in nodes:
        graph.add_node(name, function)
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()


def run_pipeline(
    notes: list[Note],
    criteria: list[Criterion],
    settings: Settings,
    provider: Provider,
    on_progress: Callable[[str], None] | None = None,
) -> RunResult:
    validate_notes(notes)
    if not criteria:
        raise ValueError("Provide at least one trial criterion.")
    keys = [(criterion.trial_name, criterion.name) for criterion in criteria]
    if len(keys) != len(set(keys)):
        raise ValueError("Criterion names must be unique within each trial.")
    run_id = uuid4().hex
    output_dir = settings.output_dir / "runs" / run_id
    output_dir.mkdir(parents=True)
    write_json(output_dir / "run.json", {"run_id": run_id, "status": "running"})
    state = PipelineState(
        notes=notes, criteria=criteria, summaries=[], eligibility_results=[], evaluation_results=[]
    )
    try:
        store = SummaryStore(settings.output_dir / "clinical_data.db")
        index = ChunkIndex(provider, output_dir / "chromadb", settings)
        graph = build_workflow(provider, index, store, run_id)
        stages = []
        for update in graph.stream(state, stream_mode="updates"):
            for stage, values in update.items():
                state.update(values or {})
                stages.append(stage)
                logger.info("Run %s completed stage %s", run_id, stage)
                if on_progress:
                    on_progress(stage)
        result = RunResult(
            run_id,
            output_dir,
            state,
            [chunk.model_dump() for chunk in index.chunks],
            store.rows(run_id),
            stages,
        )
        write_json(
            output_dir / "summaries.json", [item.model_dump() for item in state["summaries"]]
        )
        write_json(output_dir / "criteria.json", [item.model_dump() for item in state["criteria"]])
        write_json(
            output_dir / "eligibility.json",
            [item.model_dump() for item in state["eligibility_results"]],
        )
        write_json(output_dir / "chunks.json", result.chunks)
        write_csv(result.csv_path, state["evaluation_results"], COLUMNS)
        for trial_name, criterion_name in keys:
            rows = [
                row
                for row in state["evaluation_results"]
                if row["Trial_Name"] == trial_name and row["Criteria_Name"] == criterion_name
            ]
            write_csv(
                output_dir
                / "evaluation"
                / filename(trial_name)
                / (filename(criterion_name) + ".csv"),
                rows,
                COLUMNS,
            )
        write_json(
            output_dir / "run.json",
            {
                "run_id": run_id,
                "status": "completed_with_errors" if result.has_errors else "completed",
                "stages": stages,
                "patients": len(notes),
                "criteria": len(criteria),
                "model": settings.model,
                "embedding_model": settings.embedding_model,
                "summary_model": settings.summary_model or settings.model,
                "provider": type(provider).__name__,
            },
        )
        with ZipFile(output_dir / "report.zip", "w") as archive:
            for path in sorted(output_dir.rglob("*")):
                relative = path.relative_to(output_dir)
                if (
                    path.is_file()
                    and "chromadb" not in relative.parts
                    and path.name != "report.zip"
                ):
                    archive.write(path, relative.as_posix())
        return result
    except Exception as error:
        write_json(
            output_dir / "run.json",
            {
                "run_id": run_id,
                "status": "failed",
                "error_type": type(error).__name__,
            },
        )
        raise
