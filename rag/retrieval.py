"""Retrieval over the corpus.

Honest note on what this is: the project brief asks for retrieval over
PostgreSQL + pgvector. There is no Postgres instance available in this
sandbox, so `VectorIndex` is the interface both a real pgvector-backed
index and the local fallback implement, and `TfidfVectorIndex` (numpy +
scikit-learn TF-IDF cosine similarity, persisted to a small file-backed
index) is the fallback actually used by the app, tests, and CI here.
`PgVectorIndex` below is a documented stub showing the exact swap-in
shape -- it is not implemented or tested, because doing so needs a live
database this sandbox doesn't have. See README "What is implemented /
what is not".

Everything above the `VectorIndex` interface (the LangGraph pipeline,
the MCP server, the API) only calls `search()` and `get_chunk()`, so
swapping `TfidfVectorIndex` for a real `PgVectorIndex` is a one-line
change in `rag/pipeline.py` / `mcp_server/server.py`.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = ROOT / "corpus" / "data" / "chunks.jsonl"
DEFAULT_INDEX_PATH = ROOT / "rag" / "index_cache" / "tfidf_index.pkl"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    source_url: str
    chapter: str
    section_number: str
    section_title: str
    score: float


class VectorIndex(Protocol):
    """Interface any retrieval backend (local TF-IDF, pgvector, ...) implements."""

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...

    def get_chunk(self, chunk_id: str) -> dict | None: ...


def load_chunks(path: Path = DEFAULT_CHUNKS_PATH) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class TfidfVectorIndex:
    """In-memory / file-backed TF-IDF cosine-similarity index.

    This is the local dev/test/CI fallback for pgvector: no embedding
    model or database required, fully deterministic (same query always
    gets the same ranked results, which matters for a reproducible eval
    gate), and small enough to commit a cached copy if desired. Real
    semantic embeddings (e.g. an Anthropic/OpenAI embedding endpoint, or
    pgvector with a local sentence-transformers model) would improve
    recall on paraphrased queries; TF-IDF only matches on shared
    vocabulary. That tradeoff is deliberate here given no LLM/embedding
    API key is available in this sandbox -- see README.
    """

    def __init__(self, chunks: list[dict] | None = None, chunks_path: Path = DEFAULT_CHUNKS_PATH):
        self.chunks = chunks if chunks is not None else load_chunks(chunks_path)
        self._by_id = {c["chunk_id"]: c for c in self.chunks}
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
        )
        texts = [c["text"] for c in self.chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0.0:
                continue
            c = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    document_title=c["document_title"],
                    text=c["text"],
                    source_url=c["source_url"],
                    chapter=c.get("chapter", ""),
                    section_number=c.get("section_number", ""),
                    section_title=c.get("section_title", ""),
                    score=round(score, 4),
                )
            )
        return results

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._by_id.get(chunk_id)

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectorizer": self._vectorizer, "matrix": self._matrix}, f)

    @classmethod
    def load(cls, path: Path = DEFAULT_INDEX_PATH) -> "TfidfVectorIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks = data["chunks"]
        obj._by_id = {c["chunk_id"]: c for c in obj.chunks}
        obj._vectorizer = data["vectorizer"]
        obj._matrix = data["matrix"]
        return obj


class PgVectorIndex:
    """Documented stub for a real PostgreSQL + pgvector backend.

    NOT implemented or tested -- this sandbox has no Postgres instance
    and no embedding-model API key. It exists to show the intended
    production shape and satisfy the same `VectorIndex` interface as
    `TfidfVectorIndex`, so swapping it in is mechanical once both a
    database and an embedding endpoint are available:

        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE chunks (
            chunk_id text PRIMARY KEY,
            document_id text NOT NULL,
            document_title text NOT NULL,
            text text NOT NULL,
            source_url text NOT NULL,
            chapter text, section_number text, section_title text,
            embedding vector(1536)
        );
        CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);

    `search()` would embed the query with the same model used at ingest
    time, then `SELECT ... ORDER BY embedding <=> :query_embedding LIMIT :k`.
    """

    def __init__(self, dsn: str, embed_fn=None):
        raise NotImplementedError(
            "PgVectorIndex is a documented interface stub, not a working "
            "backend -- no Postgres/pgvector instance or embedding API key "
            "is available in this sandbox. Use TfidfVectorIndex."
        )
