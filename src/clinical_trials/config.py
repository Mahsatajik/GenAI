"""Explicit configuration shared by the CLI, UI, and Python API."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    output_dir: Path = Path("outputs")
    chunk_size: int = 450
    chunk_overlap: int = 50
    top_k: int = 5
    model: str = ""
    embedding_model: str = ""
    summary_model: str = ""

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Require 0 <= chunk_overlap < chunk_size.")
        if self.top_k < 1:
            raise ValueError("top_k must be positive.")

    @classmethod
    def from_env(cls, **overrides):
        values = {
            "model": os.getenv("GEMINI_MODEL", ""),
            "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", ""),
            "summary_model": os.getenv("GEMINI_SUMMARY_MODEL", ""),
        }
        values.update(overrides)
        return cls(**values)
