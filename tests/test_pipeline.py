"""Tests for the LangGraph answering pipeline. Uses a small fake index and
a scripted fake generator (no LLM, no network) so the graph wiring itself
-- retrieve -> generate -> verify -> finalize, and both refusal gates --
is exercised deterministically.
"""

from __future__ import annotations

from rag.pipeline import AnsweringPipeline, NOT_SUPPORTED_MESSAGE
from rag.retrieval import RetrievedChunk
from rag.verification import Claim


class FakeIndex:
    """Deterministic stand-in for TfidfVectorIndex."""

    def __init__(self, results: list[RetrievedChunk], chunks_by_id: dict[str, dict]):
        self._results = results
        self._chunks_by_id = chunks_by_id

    def search(self, query: str, k: int = 5):
        return self._results[:k]

    def get_chunk(self, chunk_id: str):
        return self._chunks_by_id.get(chunk_id)


class ScriptedGenerator:
    """Always returns a fixed set of claims, ignoring the question."""

    model_id = "scripted-test-model"

    def __init__(self, claims: list[Claim]):
        self._claims = claims

    def generate(self, question, chunks):
        from rag.generation import GenerationOutput

        return GenerationOutput(claims=self._claims, model_id=self.model_id, raw_text="scripted")


def _chunk(chunk_id, text, score):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        document_title="Doc",
        text=text,
        source_url="https://example.org",
        chapter="",
        section_number="",
        section_title="",
        score=score,
    )


def test_pipeline_answers_when_claims_are_grounded():
    chunk = _chunk("c1", "Everyone is equal before the law.", score=0.5)
    index = FakeIndex([chunk], {"c1": {"text": chunk.text}})
    generator = ScriptedGenerator([Claim(text="Everyone is equal before the law.", chunk_id="c1")])

    pipeline = AnsweringPipeline(index=index, generator=generator)
    state = pipeline.answer("Is everyone equal before the law?")

    assert state["answered"] is True
    assert state["refusal_reason"] is None
    assert "[c1]" in state["answer_text"]
    assert state["transparency"].answered is True
    assert state["transparency"].model_id == "scripted-test-model"


def test_pipeline_refuses_on_low_retrieval_relevance():
    chunk = _chunk("c1", "Irrelevant text.", score=0.01)  # below RETRIEVAL_SCORE_THRESHOLD
    index = FakeIndex([chunk], {"c1": {"text": chunk.text}})
    generator = ScriptedGenerator([Claim(text="Should never be used.", chunk_id="c1")])

    pipeline = AnsweringPipeline(index=index, generator=generator)
    state = pipeline.answer("Unrelated question?")

    assert state["answered"] is False
    assert state["refusal_reason"] == "no_relevant_chunks_retrieved"
    assert state["answer_text"] == NOT_SUPPORTED_MESSAGE
    # Generator must not have been consulted -- no claims in state.
    assert state["claims"] == []


def test_pipeline_refuses_when_claims_fail_verification():
    chunk = _chunk("c1", "The Bank of Finland operates under Parliament.", score=0.5)
    index = FakeIndex([chunk], {"c1": {"text": chunk.text}})
    # Claim text has nothing to do with the chunk it cites -> verification drops it.
    generator = ScriptedGenerator([Claim(text="Bananas grow well in tropical climates.", chunk_id="c1")])

    pipeline = AnsweringPipeline(index=index, generator=generator)
    state = pipeline.answer("Some question?")

    assert state["answered"] is False
    assert state["refusal_reason"] == "no_claims_survived_verification"
    assert state["answer_text"] == NOT_SUPPORTED_MESSAGE


def test_pipeline_drops_unsupported_claims_but_keeps_supported_ones():
    chunk = _chunk("c1", "Everyone is equal before the law. The sky was painted purple by wizards.", score=0.5)
    index = FakeIndex([chunk], {"c1": {"text": chunk.text}})
    generator = ScriptedGenerator(
        [
            Claim(text="Everyone is equal before the law.", chunk_id="c1"),  # supported
            Claim(text="The sky was painted purple by wizards riding dragons over the ocean.", chunk_id="c1"),  # not
        ]
    )

    pipeline = AnsweringPipeline(index=index, generator=generator)
    state = pipeline.answer("Tell me things.")

    assert state["answered"] is True
    assert "[c1]" in state["answer_text"]
    assert "wizards riding dragons over the ocean" not in state["answer_text"]
    # transparency records the verification outcome of BOTH claims, even the dropped one
    reasons = {v["supported"] for v in state["transparency"].verification}
    assert reasons == {True, False}


def test_pipeline_transparency_record_has_required_fields():
    chunk = _chunk("c1", "Everyone is equal before the law.", score=0.5)
    index = FakeIndex([chunk], {"c1": {"text": chunk.text}})
    generator = ScriptedGenerator([Claim(text="Everyone is equal before the law.", chunk_id="c1")])

    pipeline = AnsweringPipeline(index=index, generator=generator)
    state = pipeline.answer("Is everyone equal before the law?")
    record = state["transparency"].to_json()

    for field in ("model_id", "prompt_hash", "retrieved_chunks", "verification", "answered", "timestamp_utc"):
        assert field in record
    assert record["timestamp_utc"].endswith("Z")
    assert record["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert "similarity_score" in record["retrieved_chunks"][0]
