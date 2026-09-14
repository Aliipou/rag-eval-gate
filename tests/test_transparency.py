"""Tests for the transparency record builder."""

from __future__ import annotations

import json

from rag.retrieval import RetrievedChunk
from rag.transparency import TransparencyRecord
from rag.verification import VerificationResult


def _chunk(chunk_id, score):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        document_title="Doc",
        text="text",
        source_url="https://example.org",
        chapter="",
        section_number="",
        section_title="",
        score=score,
    )


def test_build_includes_all_required_fields():
    retrieved = [_chunk("c1", 0.42)]
    verification = [VerificationResult(claim_text="x", chunk_id="c1", supported=True, reason="lexical_coverage_ok", coverage=0.8)]
    record = TransparencyRecord.build(
        model_id="model-x",
        prompt_hash="abc123",
        question="q?",
        retrieved=retrieved,
        verification_results=verification,
        answered=True,
        refusal_reason=None,
    )
    data = record.to_json()
    assert data["model_id"] == "model-x"
    assert data["prompt_hash"] == "abc123"
    assert data["question"] == "q?"
    assert data["retrieved_chunks"] == [{"chunk_id": "c1", "document_id": "d1", "similarity_score": 0.42}]
    assert data["verification"] == [
        {"claim_text": "x", "chunk_id": "c1", "supported": True, "reason": "lexical_coverage_ok", "coverage": 0.8}
    ]
    assert data["answered"] is True
    assert data["refusal_reason"] is None
    assert "timestamp_utc" in data


def test_timestamp_is_utc_iso_format():
    record = TransparencyRecord.build(
        model_id="m", prompt_hash="h", question="q", retrieved=[], verification_results=[],
        answered=False, refusal_reason="no_relevant_chunks_retrieved",
    )
    ts = record.to_json()["timestamp_utc"]
    assert ts.endswith("Z")
    assert "T" in ts


def test_to_json_str_is_valid_json():
    record = TransparencyRecord.build(
        model_id="m", prompt_hash="h", question="q", retrieved=[], verification_results=[],
        answered=False, refusal_reason="no_claims_survived_verification",
    )
    parsed = json.loads(record.to_json_str())
    assert parsed["refusal_reason"] == "no_claims_survived_verification"


def test_refused_record_has_no_answered_claims_but_keeps_verification_trail():
    verification = [VerificationResult(claim_text="x", chunk_id="c1", supported=False, reason="lexical_coverage_below_threshold", coverage=0.1)]
    record = TransparencyRecord.build(
        model_id="m", prompt_hash="h", question="q", retrieved=[_chunk("c1", 0.2)],
        verification_results=verification, answered=False, refusal_reason="no_claims_survived_verification",
    )
    data = record.to_json()
    assert data["answered"] is False
    assert data["verification"][0]["supported"] is False
