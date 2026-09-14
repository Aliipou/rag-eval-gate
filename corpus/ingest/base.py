"""Shared types for corpus ingestion.

An `Ingester` turns one source document (a PDF, an HTML page, ...) into a
list of `Chunk` objects that get written to `corpus/data/chunks.jsonl`.
Each ingester is independent and pluggable: `build_corpus.py` just calls
`ingest()` on every registered source and concatenates the results, so
adding a new corpus (a different Finlex statute, a different ministry
PDF, a live database dump, ...) means writing one new module with an
`ingest(source) -> list[Chunk]` function and registering it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Chunk:
    """One retrievable unit of the corpus.

    `chunk_id` must be stable and globally unique across the whole
    corpus -- it is what citations, the golden set, and the transparency
    record all point at.
    """

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    source_url: str
    licence: str
    retrieved_at: str
    chapter: str = ""
    section_number: str = ""
    section_title: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
