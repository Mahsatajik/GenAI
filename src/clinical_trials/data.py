"""Load the original notebook CSV formats without a pandas dependency."""

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import Criterion, Note


def read_notes(path: Path) -> list[Note]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("Notes CSV must contain a 'text' column.")
        notes = []
        for index, row in enumerate(reader, start=1):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            patient_id = (row.get("patient_id") or f"patient-{index:04d}").strip()
            notes.append(Note(patient_id=patient_id, text=text))
    validate_notes(notes)
    return notes


def validate_notes(notes: list[Note]) -> None:
    if not notes:
        raise ValueError("Provide at least one nonempty clinical note.")
    ids = [note.patient_id for note in notes]
    if len(ids) != len(set(ids)):
        raise ValueError("Patient IDs must be unique within a run.")


def read_criteria(source: Path | list[Path]) -> list[Criterion]:
    paths = sorted(source) if isinstance(source, list) else sorted(Path(source).glob("*.csv"))
    if not paths:
        raise ValueError("Provide at least one trial criteria CSV file.")
    criteria = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"Criteria_name", "Content"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path.name} needs 'Criteria_name' and 'Content' columns.")
            names = set()
            for index, row in enumerate(reader, start=2):
                name = (row.get("Criteria_name") or "").strip()
                text = (row.get("Content") or "").strip()
                if not name or not text:
                    raise ValueError(
                        f"{path.name}, row {index}: criterion name and content are required."
                    )
                if name in names:
                    raise ValueError(f"Duplicate criterion {name!r} in {path.name}.")
                names.add(name)
                prefix = name.lower().split("_", 1)[0]
                kind = prefix if prefix in {"inclusion", "exclusion"} else "unspecified"
                criteria.append(Criterion(trial_name=path.stem, name=name, text=text, kind=kind))
    if not criteria:
        raise ValueError("Trial criteria files contain no criteria.")
    return criteria


def write_csv(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
