"""FastAPI backend for the RAG answering service.

    POST /api/answer                  -> ask a question, get an answer +
                                          inline transparency record
    GET  /api/transparency/{prompt_hash} -> re-fetch a transparency
                                          record as downloadable JSON
    GET  /api/health

The transparency record is always returned inline in the /api/answer
response (that's what "exportable as JSON from the API" means at
minimum). /api/transparency/{prompt_hash} additionally lets the UI (or
anyone) re-fetch/download the same record later by the hash printed in
the original response, backed by a small in-memory cache of the most
recent answers -- NOT a persistent store. Restarting the process, or
asking more than `MAX_CACHE` questions, drops old entries. A real
deployment would persist these (e.g. the same Postgres that would hold
pgvector -- see rag/retrieval.py) instead of keeping them in memory;
that's out of scope for this sandbox (see README).
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.pipeline import AnsweringPipeline

MAX_CACHE = 200

app = FastAPI(
    title="rag-eval-gate API",
    description=(
        "Retrieval-augmented answering over a small real Finnish-law "
        "corpus, with an explicit transparency record on every answer. "
        "See the repository README for what is and isn't implemented."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # portfolio/demo project, not a production deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: AnsweringPipeline | None = None
_transparency_cache: "OrderedDict[str, dict]" = OrderedDict()


def get_pipeline() -> AnsweringPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnsweringPipeline()
    return _pipeline


class AnswerRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    text: str


class AnswerResponse(BaseModel):
    question: str
    answer_text: str
    answered: bool
    refusal_reason: str | None
    citations: list[Citation]
    transparency: dict


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    pipeline = get_pipeline()
    state = pipeline.answer(req.question)

    verified_claims = state.get("verified_claims", [])
    chunks_by_id = {c.chunk_id: c for c in state.get("retrieved", [])}
    citations = [
        Citation(chunk_id=c.chunk_id, text=chunks_by_id[c.chunk_id].text)
        for c in verified_claims
        if c.chunk_id in chunks_by_id
    ]

    transparency = state["transparency"].to_json()
    _transparency_cache[transparency["prompt_hash"]] = transparency
    while len(_transparency_cache) > MAX_CACHE:
        _transparency_cache.popitem(last=False)

    return AnswerResponse(
        question=req.question,
        answer_text=state["answer_text"],
        answered=state["answered"],
        refusal_reason=state.get("refusal_reason"),
        citations=citations,
        transparency=transparency,
    )


@app.get("/api/transparency/{prompt_hash}")
def get_transparency(prompt_hash: str) -> dict:
    record = _transparency_cache.get(prompt_hash)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No cached transparency record for this prompt_hash "
                "(the in-memory cache is not persistent -- see module docstring)."
            ),
        )
    return record
