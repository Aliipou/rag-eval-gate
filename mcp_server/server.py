"""MCP server exposing the corpus over two tools, via FastMCP.

    search_corpus(query, k=5) -> ranked chunks with similarity scores
    get_chunk(chunk_id)       -> a single chunk's full text + metadata

Backed by the same `TfidfVectorIndex` the answering pipeline uses (see
rag/retrieval.py for what that is and isn't -- the local pgvector
fallback). Any MCP client (Claude Desktop, another agent, a test script)
can use this to retrieve grounded source text without going through the
answering pipeline at all.

Run with:
    python -m mcp_server.server              # stdio transport
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import FastMCP

from rag.retrieval import TfidfVectorIndex

mcp = FastMCP(
    name="rag-eval-gate-corpus",
    instructions=(
        "Search and fetch chunks from the rag-eval-gate corpus: the "
        "Constitution of Finland (731/1999, unofficial Finlex "
        "translation) and a Ministry of Economic Affairs and Employment "
        "guide to the Employment Contracts Act. See corpus/SOURCES.md "
        "in the repository for full source/licence details."
    ),
)

_index: TfidfVectorIndex | None = None


def get_index() -> TfidfVectorIndex:
    global _index
    if _index is None:
        _index = TfidfVectorIndex()
    return _index


@mcp.tool
def search_corpus(query: str, k: int = 5) -> list[dict]:
    """Search the corpus for chunks relevant to `query`.

    Returns up to `k` chunks ranked by TF-IDF cosine similarity
    (descending), each with its chunk_id, document_id, document_title,
    section info, similarity score, and full chunk text.
    """
    results = get_index().search(query, k=k)
    return [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "document_title": r.document_title,
            "chapter": r.chapter,
            "section_number": r.section_number,
            "section_title": r.section_title,
            "similarity_score": r.score,
            "text": r.text,
        }
        for r in results
    ]


@mcp.tool
def get_chunk(chunk_id: str) -> dict:
    """Fetch a single chunk by its exact chunk_id.

    Returns an object with `found: false` if no chunk with that id
    exists in the corpus, rather than raising -- callers (including
    other LLM agents) can branch on `found` without needing try/except.
    """
    chunk = get_index().get_chunk(chunk_id)
    if chunk is None:
        return {"found": False, "chunk_id": chunk_id}
    return {"found": True, **chunk}


if __name__ == "__main__":
    mcp.run()
