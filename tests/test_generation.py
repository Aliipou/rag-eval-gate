"""Tests for claim generation. No live LLM call is made anywhere here --
`ExtractiveGenerator` is deterministic and API-free, and `AnthropicGenerator`
is tested with the `anthropic` client mocked out.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from rag.generation import AnthropicGenerator, ExtractiveGenerator, prompt_hash
from rag.retrieval import RetrievedChunk


def _chunk(chunk_id, text, score=0.5, document_id="doc1"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title="Doc",
        text=text,
        source_url="https://example.org",
        chapter="",
        section_number="",
        section_title="",
        score=score,
    )


class TestExtractiveGenerator:
    def test_generates_claims_cited_to_retrieved_chunks_only(self):
        chunks = [
            _chunk("a", "Everyone is equal before the law. No one shall be discriminated against.", score=0.4),
            _chunk("b", "The Bank of Finland operates under Parliament.", score=0.1),
        ]
        gen = ExtractiveGenerator()
        out = gen.generate("Is everyone equal before the law?", chunks)
        assert out.claims, "expected at least one claim"
        cited_ids = {c.chunk_id for c in out.claims}
        assert cited_ids <= {"a", "b"}
        assert out.model_id == "extractive-lexical-v1"

    def test_no_overlap_produces_no_claims(self):
        chunks = [_chunk("a", "The weather in Helsinki is often cold in winter.", score=0.2)]
        gen = ExtractiveGenerator()
        out = gen.generate("What is the population of Japan?", chunks)
        assert out.claims == []
        assert out.raw_text == ""

    def test_empty_chunk_list(self):
        gen = ExtractiveGenerator()
        out = gen.generate("Any question?", [])
        assert out.claims == []

    def test_respects_max_claims(self):
        long_text = " ".join(f"Statement number {i} about equality and law." for i in range(20))
        chunks = [_chunk("a", long_text, score=0.5)]
        gen = ExtractiveGenerator(max_claims=2)
        out = gen.generate("Tell me about equality and law", chunks)
        assert len(out.claims) <= 2

    def test_high_confidence_chunk_allows_single_word_overlap(self):
        # Chunk score above the high-confidence bar: a sentence sharing
        # just one significant word with the question is still eligible.
        chunks = [_chunk("a", "The property of everyone is safeguarded.", score=0.9)]
        gen = ExtractiveGenerator(min_overlap=2, high_confidence_chunk_score=0.3)
        out = gen.generate("Is property protected?", chunks)
        assert len(out.claims) == 1

    def test_low_confidence_chunk_requires_full_min_overlap(self):
        chunks = [_chunk("a", "The property of everyone is safeguarded.", score=0.05)]
        gen = ExtractiveGenerator(min_overlap=2, high_confidence_chunk_score=0.3)
        out = gen.generate("Is property protected?", chunks)
        assert out.claims == []

    def test_no_duplicate_sentences(self):
        chunks = [
            _chunk("a", "Everyone has the right to equality.", score=0.6),
            _chunk("b", "Everyone has the right to equality.", score=0.5),
        ]
        gen = ExtractiveGenerator()
        out = gen.generate("What right does everyone have to equality?", chunks)
        texts = [c.text for c in out.claims]
        assert len(texts) == len(set(texts))


def test_prompt_hash_stable_and_sensitive_to_inputs():
    chunks = [_chunk("a", "text", score=0.5)]
    h1 = prompt_hash("question?", chunks, "model-x")
    h2 = prompt_hash("question?", chunks, "model-x")
    assert h1 == h2
    assert h1 != prompt_hash("different question?", chunks, "model-x")
    assert h1 != prompt_hash("question?", chunks, "model-y")


class TestAnthropicGenerator:
    def test_parses_json_claims_from_mocked_client(self):
        fake_response = MagicMock()
        fake_block = MagicMock()
        fake_block.type = "text"
        fake_block.text = json.dumps(
            [{"text": "Everyone is equal before the law.", "chunk_id": "c1"}]
        )
        fake_response.content = [fake_block]

        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        with patch("anthropic.Anthropic", return_value=fake_client):
            gen = AnthropicGenerator(model_id="claude-test", api_key="fake-key-not-used")
            chunks = [_chunk("c1", "Everyone is equal before the law.", score=0.5)]
            out = gen.generate("Is everyone equal?", chunks)

        assert out.claims == [__import__("rag.verification", fromlist=["Claim"]).Claim(
            text="Everyone is equal before the law.", chunk_id="c1"
        )]
        assert out.model_id == "claude-test"
        fake_client.messages.create.assert_called_once()

    def test_malformed_json_yields_no_claims(self):
        fake_response = MagicMock()
        fake_block = MagicMock()
        fake_block.type = "text"
        fake_block.text = "not valid json"
        fake_response.content = [fake_block]

        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        with patch("anthropic.Anthropic", return_value=fake_client):
            gen = AnthropicGenerator(api_key="fake-key-not-used")
            out = gen.generate("question?", [_chunk("c1", "text", score=0.5)])

        assert out.claims == []
