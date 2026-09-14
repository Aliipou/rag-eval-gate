"""Tests for the TF-IDF retrieval fallback. Uses a small in-memory chunk
list (not the full corpus) so tests run fast and don't depend on corpus
content -- corpus-shape regressions are caught by eval/run_eval.py instead.
"""

from __future__ import annotations

import pytest

from rag.retrieval import TfidfVectorIndex

SAMPLE_CHUNKS = [
    {
        "chunk_id": "c1",
        "document_id": "d1",
        "document_title": "Doc 1",
        "text": "Everyone is equal before the law in Finland.",
        "source_url": "https://example.org/1",
        "chapter": "",
        "section_number": "1",
        "section_title": "Equality",
    },
    {
        "chunk_id": "c2",
        "document_id": "d1",
        "document_title": "Doc 1",
        "text": "The Bank of Finland operates under the guarantee of Parliament.",
        "source_url": "https://example.org/2",
        "chapter": "",
        "section_number": "2",
        "section_title": "Bank",
    },
    {
        "chunk_id": "c3",
        "document_id": "d2",
        "document_title": "Doc 2",
        "text": "Trial periods in employment contracts last up to four months.",
        "source_url": "https://example.org/3",
        "chapter": "",
        "section_number": "",
        "section_title": "",
    },
]


@pytest.fixture
def index():
    return TfidfVectorIndex(chunks=SAMPLE_CHUNKS)


def test_search_returns_relevant_chunk_first(index):
    results = index.search("Is everyone equal before the law?", k=3)
    assert results, "expected at least one result"
    assert results[0].chunk_id == "c1"


def test_search_respects_k(index):
    results = index.search("Finland", k=1)
    assert len(results) <= 1


def test_search_empty_query_returns_empty(index):
    assert index.search("", k=3) == []
    assert index.search("   ", k=3) == []


def test_search_unrelated_query_scores_low_or_empty(index):
    results = index.search("photosynthesis in tropical rainforests", k=3)
    for r in results:
        assert r.score < 0.2


def test_get_chunk_by_id(index):
    chunk = index.get_chunk("c3")
    assert chunk is not None
    assert chunk["document_id"] == "d2"


def test_get_chunk_unknown_id_returns_none(index):
    assert index.get_chunk("does-not-exist") is None


def test_scores_are_sorted_descending(index):
    results = index.search("Finland Bank Parliament equal law", k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_save_and_load_roundtrip(tmp_path, index):
    path = tmp_path / "index.pkl"
    index.save(path)
    loaded = TfidfVectorIndex.load(path)
    r1 = index.search("equal before the law", k=2)
    r2 = loaded.search("equal before the law", k=2)
    assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
