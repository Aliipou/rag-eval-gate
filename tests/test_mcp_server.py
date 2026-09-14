"""Tests for the FastMCP server's two tools, exercised in-process via
fastmcp's Client (no subprocess, no network) against the real committed
corpus."""

from __future__ import annotations

import pytest
from fastmcp import Client

from mcp_server.server import mcp


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


@pytest.mark.asyncio
async def test_lists_both_tools(client):
    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert names == {"search_corpus", "get_chunk"}


@pytest.mark.asyncio
async def test_search_corpus_returns_ranked_results(client):
    result = await client.call_tool("search_corpus", {"query": "Is everyone equal before the law?", "k": 3})
    data = result.data
    assert isinstance(data, list)
    assert len(data) <= 3
    assert data[0]["chunk_id"] == "const-731-1999-s6"
    scores = [r["similarity_score"] for r in data]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_get_chunk_found(client):
    result = await client.call_tool("get_chunk", {"chunk_id": "const-731-1999-s6"})
    data = result.data
    assert data["found"] is True
    assert data["document_id"] == "const-731-1999"
    assert "Equality" in data["section_title"]


@pytest.mark.asyncio
async def test_get_chunk_not_found(client):
    result = await client.call_tool("get_chunk", {"chunk_id": "does-not-exist"})
    assert result.data == {"found": False, "chunk_id": "does-not-exist"}
