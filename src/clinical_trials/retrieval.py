"""Patient-scoped Chroma retrieval with explicit document and query embeddings."""

from pathlib import Path

import numpy as np

from .config import Settings
from .models import Chunk, Summary
from .providers import Provider


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class ChunkIndex:
    def __init__(self, provider: Provider, path: Path, settings: Settings):
        import chromadb

        self.provider = provider
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name="summary_chunks",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self.chunks: list[Chunk] = []
        self.counts: dict[str, int] = {}
        self.dimension: int | None = None

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        values = self.provider.embed(texts, task)
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[0] != len(texts) or not array.shape[1]:
            raise ValueError("Embedding responses must contain one vector per text.")
        if not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) == 0):
            raise ValueError("Embedding vectors must be finite and nonzero.")
        if self.dimension is not None and array.shape[1] != self.dimension:
            raise ValueError("Document and query embedding dimensions must match.")
        self.dimension = array.shape[1]
        return array.tolist()

    def add(self, summaries: list[Summary]) -> None:
        chunks = []
        for summary in summaries:
            if summary.error:
                continue
            texts = split_text(
                summary.summary, self.settings.chunk_size, self.settings.chunk_overlap
            )
            self.counts[summary.patient_id] = len(texts)
            chunks.extend(
                Chunk(
                    chunk_id=f"{summary.patient_id}:{i}",
                    patient_id=summary.patient_id,
                    document=text,
                )
                for i, text in enumerate(texts)
            )
        # Keep request payloads bounded for larger datasets.
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            embeddings = self.embed([chunk.document for chunk in batch], "RETRIEVAL_DOCUMENT")
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.document for chunk in batch],
                metadatas=[{"patient_id": chunk.patient_id} for chunk in batch],
                embeddings=embeddings,
            )
        self.chunks = chunks

    def query(self, patient_id: str, text: str, limit: int | None = None) -> list[Chunk]:
        count = min(self.counts.get(patient_id, 0), limit or self.settings.top_k)
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.embed([text], "RETRIEVAL_QUERY"),
            where={"patient_id": patient_id},
            n_results=count,
            include=["documents"],
        )
        return [
            Chunk(chunk_id=chunk_id, patient_id=patient_id, document=document)
            for chunk_id, document in zip(result["ids"][0], result["documents"][0])
        ]
