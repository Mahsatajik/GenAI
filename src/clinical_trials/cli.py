"""Command-line entry points with explicit inputs and lazy UI loading."""

import argparse
from importlib.resources import files
import logging
from pathlib import Path
import sys

from .config import Settings
from .data import read_criteria, read_notes
from .pipeline import run_pipeline
from .providers import DemoProvider, GeminiProvider


def make_provider(settings: Settings, demo: bool):
    return DemoProvider() if demo else GeminiProvider(settings)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="trial-screen", description="Screen clinical notes against trial criteria."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the pipeline using Gemini.")
    run.add_argument("--notes", type=Path, required=True)
    run.add_argument("--criteria", type=Path, required=True)
    demo = subparsers.add_parser("demo", help="Run bundled synthetic examples without API calls.")
    demo.add_argument("--notes", type=Path)
    demo.add_argument("--criteria", type=Path)
    serve = subparsers.add_parser("serve", help="Launch the optional Gradio interface.")
    serve.add_argument("--criteria", type=Path)
    serve.add_argument("--demo", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    for command in (run, demo, serve):
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
        command.add_argument("--model", help="Generation model ID. Defaults to GEMINI_MODEL.")
        command.add_argument("--embedding-model", help="Embedding model ID.")
        command.add_argument("--summary-model", help="Optional existing tuned summarization model.")
        command.add_argument("--chunk-size", type=int, default=450)
        command.add_argument("--chunk-overlap", type=int, default=50)
        command.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    overrides = {
        "output_dir": args.output_dir,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
    }
    for name in ("model", "embedding_model", "summary_model"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    demo_mode = args.command == "demo" or getattr(args, "demo", False)
    resource_dir = Path(str(files("clinical_trials") / "examples"))
    criteria_dir = args.criteria or (resource_dir / "trials" if demo_mode else None)
    provider = None
    try:
        settings = Settings.from_env(**overrides)
        criteria = read_criteria(criteria_dir) if criteria_dir is not None else []
        if args.command == "serve":
            from .ui import build_app, launch_app

            launch_app(build_app(settings, criteria, demo_mode), settings, args.host, args.port)
            return 0
        notes = read_notes(args.notes or resource_dir / "notes.csv")
        provider = make_provider(settings, demo_mode)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        result = run_pipeline(notes, criteria, settings, provider)
        print(f"Run: {result.run_id}")
        print(f"Status: {'completed_with_errors' if result.has_errors else 'completed'}")
        print(f"Results: {result.csv_path}")
        return 2 if result.has_errors else 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if provider is not None:
            provider.close()
