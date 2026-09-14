"""Tests for the FastAPI backend, using FastAPI's TestClient (in-process,
no network) against the real committed corpus and the default
ExtractiveGenerator -- no LLM calls."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_answer_returns_grounded_answer_with_citations_and_transparency():
    resp = client.post("/api/answer", json={"question": "Is everyone equal before the law in Finland?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answered"] is True
    assert data["citations"], "expected at least one citation"
    for c in data["citations"]:
        assert c["chunk_id"].startswith("const-") or c["chunk_id"].startswith("emp-")
        assert c["text"]
    transparency = data["transparency"]
    for field in ("model_id", "prompt_hash", "retrieved_chunks", "verification", "timestamp_utc"):
        assert field in transparency


def test_answer_refuses_genuinely_unanswerable_question():
    resp = client.post("/api/answer", json={"question": "What is the boiling point of water?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answered"] is False
    assert data["citations"] == []
    assert data["refusal_reason"] is not None


def test_answer_rejects_empty_question():
    resp = client.post("/api/answer", json={"question": "   "})
    assert resp.status_code == 400


def test_transparency_export_roundtrip():
    ask = client.post("/api/answer", json={"question": "Is everyone equal before the law in Finland?"})
    prompt_hash = ask.json()["transparency"]["prompt_hash"]

    fetched = client.get(f"/api/transparency/{prompt_hash}")
    assert fetched.status_code == 200
    assert fetched.json()["prompt_hash"] == prompt_hash


def test_transparency_export_unknown_hash_404s():
    resp = client.get("/api/transparency/does-not-exist")
    assert resp.status_code == 404
