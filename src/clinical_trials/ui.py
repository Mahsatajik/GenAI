"""Upload, inspect, and export screening results using the shared pipeline."""

from dataclasses import replace
from importlib.resources import files
from pathlib import Path
import json

from .config import Settings
from .data import read_criteria, read_notes, validate_notes
from .models import Criterion, Note
from .pipeline import run_pipeline
from .providers import DemoProvider, GeminiProvider


def _table(rows: list[dict]) -> dict:
    headers = list(rows[0]) if rows else []
    data = [
        [
            json.dumps(row[key]) if isinstance(row[key], (list, dict)) else row[key]
            for key in headers
        ]
        for row in rows
    ]
    return {"headers": headers, "data": data}


def submission_notes(text: str, path: str | None) -> list[Note]:
    notes = []
    if path:
        source = Path(path)
        if source.suffix.lower() == ".txt":
            content = source.read_text(encoding="utf-8-sig").strip()
            if not content:
                raise ValueError("The uploaded text file is empty.")
            notes.append(Note(patient_id="uploaded-note", text=content))
        elif source.suffix.lower() == ".csv":
            notes.extend(read_notes(source))
        else:
            raise ValueError("Upload a CSV or UTF-8 TXT notes file.")
    if text and text.strip():
        notes.insert(0, Note(patient_id="manual-note", text=text.strip()))
    validate_notes(notes)
    return notes


def summary_card(patients=0, criteria=0, results=0, unknown=0):
    return (
        '<div class="run-totals">'
        f"<div><strong>{patients}</strong><span>Patients</span></div>"
        f"<div><strong>{criteria}</strong><span>Criteria</span></div>"
        f"<div><strong>{results}</strong><span>Assessments</span></div>"
        f"<div><strong>{unknown}</strong><span>Unknown</span></div>"
        "</div>"
    )


def build_app(settings: Settings, criteria: list[Criterion] | None = None, demo_mode=False):
    import gradio as gr

    default_criteria = criteria or []
    example_dir = Path(str(files("clinical_trials") / "examples"))
    example_notes = str(example_dir / "notes.csv")
    example_trials = [str(example_dir / "trials" / "sample_trial.csv")]

    def execute(
        note_text,
        note_path,
        trial_paths,
        mode,
        api_key,
        model,
        embedding_model,
        progress=gr.Progress(),
    ):
        provider = None
        try:
            notes = submission_notes(note_text, note_path)
            loaded_criteria = (
                read_criteria([Path(path) for path in trial_paths])
                if trial_paths
                else default_criteria
            )
            if not loaded_criteria:
                raise ValueError("Upload at least one trial criteria CSV.")
            live_settings = replace(
                settings, model=model.strip(), embedding_model=embedding_model.strip()
            )
            provider = (
                DemoProvider()
                if mode == "Synthetic demo"
                else GeminiProvider(live_settings, api_key=api_key.strip() or None)
            )
            stages = []

            def update(stage):
                stages.append(stage)
                progress(len(stages) / 6, desc=stage.replace("_", " ").title())

            result = run_pipeline(
                notes, loaded_criteria, live_settings, provider, on_progress=update
            )
            summaries = [item.model_dump() for item in result.state["summaries"]]
            assessments = [item.model_dump() for item in result.state["eligibility_results"]]
            review_rows = [
                {
                    "Patient": item.patient_id,
                    "Trial": item.trial_name,
                    "Criterion": item.criteria_name,
                    "Match": item.eligibility,
                    "Evidence": " | ".join(item.evidence) or "No supporting quotation",
                }
                for item in result.state["eligibility_results"]
            ]
            unknown = sum(item["eligibility"] == "Unknown" for item in assessments)
            status = "Completed with errors" if result.has_errors else "Screening complete"
            return (
                f"{status}. Run {result.run_id}",
                summary_card(len(notes), len(loaded_criteria), len(assessments), unknown),
                _table(review_rows),
                _table(summaries),
                _table(result.state["evaluation_results"]),
                _table(result.database_rows),
                _table(result.chunks),
                str(result.csv_path),
                str(result.output_dir / "report.zip"),
                "\n".join(result.stages),
            )
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except Exception as error:
            raise gr.Error(
                f"Processing failed ({type(error).__name__}). Check your connection and configuration."
            ) from error
        finally:
            if provider is not None:
                provider.close()

    with gr.Blocks(title="TrialLens | Clinical Trial Screening", analytics_enabled=False) as app:
        gr.HTML(
            '<header class="brand-bar"><div class="brand-name">'
            '<span class="brand-mark">T</span>TrialLens</div>'
            "<span>Clinical trial screening workspace</span></header>"
            '<section class="workspace-intro"><h1>From clinical notes<br>to reviewable evidence.</h1>'
            "<p>Upload patient notes and trial criteria. Review each match, inspect its evidence, "
            "and export the results.</p></section>"
        )
        with gr.Row(equal_height=False, elem_classes=["workspace"]):
            with gr.Column(scale=4, min_width=290, elem_classes=["input-panel"]):
                gr.Markdown("### Screening inputs")
                note_file = gr.File(
                    label="Patient notes",
                    file_types=[".csv", ".txt"],
                    type="filepath",
                    elem_id="patient-notes",
                )
                gr.Markdown(
                    "CSV needs a **text** column. **patient_id** is optional. "
                    "A TXT file represents one patient.",
                    elem_classes=["field-help"],
                )
                trial_files = gr.File(
                    label="Trial criteria",
                    file_count="multiple",
                    file_types=[".csv"],
                    type="filepath",
                    elem_id="trial-criteria",
                )
                gr.Markdown(
                    "Each trial CSV needs **Criteria_name** and **Content**. "
                    "The filename identifies the trial.",
                    elem_classes=["field-help"],
                )
                with gr.Accordion("Paste a note", open=False):
                    note_text = gr.Textbox(label="Clinical note", lines=5)
                mode = gr.Radio(
                    ["Synthetic demo", "Gemini API"],
                    label="Processing mode",
                    value="Synthetic demo" if demo_mode else "Gemini API",
                )
                with gr.Accordion("Gemini connection", open=False):
                    api_key = gr.Textbox(label="API key", type="password", value="")
                    model = gr.Textbox(label="Generation model ID", value=settings.model)
                    embedding_model = gr.Textbox(
                        label="Embedding model ID", value=settings.embedding_model
                    )
                    gr.Markdown(
                        "Environment credentials are used when the API key field is empty. "
                        "Gemini mode sends note content and criteria to Google."
                    )
                run_button = gr.Button("Run screening", variant="primary", elem_id="run-screening")
                sample_button = gr.Button("Load synthetic example", variant="secondary")
                gr.Markdown(
                    "Synthetic mode uses literal matching to demonstrate the workflow. "
                    "Its decisions are demonstration output.",
                    elem_classes=["field-help"],
                )
            with gr.Column(scale=7, min_width=300, elem_classes=["results-panel"]):
                gr.Markdown("### Review workspace")
                status = gr.Textbox(
                    label="Run status",
                    value="Upload files or load the synthetic example to begin.",
                    interactive=False,
                )
                totals = gr.HTML(summary_card())
                with gr.Tabs():
                    tables = []
                    for title in (
                        "Criterion matches",
                        "Summaries",
                        "Evidence diagnostics",
                        "Database",
                        "Chunks",
                    ):
                        with gr.Tab(title):
                            tables.append(
                                gr.Dataframe(
                                    label=title,
                                    interactive=False,
                                    wrap=True,
                                    row_count=0,
                                    column_count=0,
                                )
                            )
                    with gr.Tab("Run details"):
                        log_output = gr.Textbox(
                            label="Completed stages", lines=6, interactive=False
                        )
                gr.Markdown(
                    "**Yes** means the criterion is documented, **No** means explicitly contradicted, "
                    "and **Unknown** means insufficient evidence or a processing error. "
                    "An exclusion match still needs reviewer interpretation."
                )
                with gr.Row():
                    csv_download = gr.File(label="Results CSV", interactive=False)
                    zip_download = gr.File(label="Full report ZIP", interactive=False)
                gr.Markdown(
                    "Research screening prototype. Results require qualified human review.",
                    elem_classes=["review-note"],
                )
        run_button.click(
            execute,
            inputs=[note_text, note_file, trial_files, mode, api_key, model, embedding_model],
            outputs=[status, totals, *tables, csv_download, zip_download, log_output],
            api_name="screen",
        )
        sample_button.click(
            lambda: (
                "",
                example_notes,
                example_trials,
                "Synthetic demo",
                "Synthetic files loaded. Select Run screening to process them.",
            ),
            outputs=[note_text, note_file, trial_files, mode, status],
            api_name=False,
        )
    return app


def launch_app(app, settings: Settings, host="127.0.0.1", port=7860):
    import gradio as gr

    css = (files("clinical_trials") / "assets" / "app.css").read_text(encoding="utf-8")
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    )
    return app.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=False,
        theme=theme,
        css=css,
        allowed_paths=[str(settings.output_dir.resolve())],
        footer_links=[],
    )
