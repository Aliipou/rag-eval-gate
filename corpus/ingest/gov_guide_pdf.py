"""Ingester for plain government-published guide PDFs (page-granularity).

Unlike `finlex_statute.py`, these documents (e.g. the Finnish Ministry of
Economic Affairs and Employment's guide to the Employment Contracts Act)
are not laid out as numbered chapters/sections, so we chunk at page
granularity instead: one Chunk per PDF page, after stripping the running
header/footer page numbers PDF text extraction tends to leave behind.

This is a legitimate, common chunking strategy for prose documents
without a reliable heading grammar -- it trades citation precision
(a whole page vs. a single statute section) for robustness across
documents whose structure isn't known ahead of time.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import requests
from pypdf import PdfReader

from .base import Chunk, utc_today

_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,3}\s*$")
_PAGE_NUM_PAIR = re.compile(r"^\s*\d{1,3}\s+\d{1,3}\s*$")  # "22 23" running header


@dataclass
class GovGuideSource:
    document_id: str
    document_title: str
    source_url: str
    licence: str
    local_pdf_path: str | None = None
    min_chars: int = 80  # skip near-empty pages (covers, blank separators)


def fetch_pdf_bytes(url: str, timeout: int = 30) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower():
        raise RuntimeError(f"Expected a PDF from {url}, got Content-Type={content_type!r}")
    return resp.content


def _clean_page(text: str) -> str:
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if _PAGE_NUM_LINE.match(line) or _PAGE_NUM_PAIR.match(line):
            continue
        lines.append(line)
    return " ".join(lines).strip()


def ingest(source: GovGuideSource) -> list[Chunk]:
    if source.local_pdf_path:
        with open(source.local_pdf_path, "rb") as f:
            pdf_bytes = f.read()
    else:
        pdf_bytes = fetch_pdf_bytes(source.source_url)

    reader = PdfReader(io.BytesIO(pdf_bytes))
    today = utc_today()
    chunks: list[Chunk] = []
    for i, page in enumerate(reader.pages, start=1):
        cleaned = _clean_page(page.extract_text() or "")
        if len(cleaned) < source.min_chars:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{source.document_id}-p{i}",
                document_id=source.document_id,
                document_title=source.document_title,
                text=cleaned,
                source_url=source.source_url,
                licence=source.licence,
                retrieved_at=today,
                extra={"page": i},
            )
        )
    return chunks
