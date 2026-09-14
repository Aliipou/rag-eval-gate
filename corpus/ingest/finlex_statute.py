"""Ingester for Finlex-style statute PDFs (Chapter N / Section N layout).

Finlex's "Unofficial translations of statutes" PDFs (e.g. the Constitution
of Finland, 731/1999) follow a consistent layout:

    Chapter <N>
    <chapter title>

    Section <N>
    <section title>

    <body paragraphs...>

This module parses that layout into one Chunk per section, which gives
precise, citable, stable chunk ids (e.g. `const-731-1999-s6`) instead of
arbitrary character-offset windows.

Live fetching: `fetch_pdf_bytes()` below hits the real Finlex media API.
It needs a browser-like `User-Agent` and `Accept: application/pdf` header
-- Finlex's edge (CloudFront + a Next.js app) otherwise serves a
placeholder `image/x-png` response instead of the PDF, which is a known
quirk documented here rather than silently swallowed. To point this
ingester at a different statute, resolve the numeric "media id" from the
statute's `finlex.fi/en/legislation/translations/<year>/eng/<number>`
page (the id appears in the "Download PDF" link) and pass it in.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import requests
from pypdf import PdfReader

from .base import Chunk, utc_today

FINLEX_MEDIA_URL = (
    "https://www.finlex.fi/api/media/statute-foreign-language-translation/"
    "{media_id}/mainPdf/main.pdf"
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}

_SECTION_RE = re.compile(r"^Section\s+(\d+[a-z]?)\s*$")
_CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s*$")


@dataclass
class FinlexStatuteSource:
    document_id: str
    document_title: str
    finlex_page_url: str
    licence: str
    media_id: str | None = None  # required only for live fetch
    local_pdf_path: str | None = None  # used instead of live fetch when set


def fetch_pdf_bytes(media_id: str, timeout: int = 30) -> bytes:
    """Fetch a statute translation PDF live from Finlex.

    Raises `RuntimeError` if Finlex responds with something other than a
    PDF (this happens if the media id is wrong or stale -- Finlex then
    serves an `image/x-png` placeholder with HTTP 200, so status code
    alone is not a reliable success signal).
    """
    url = FINLEX_MEDIA_URL.format(media_id=media_id)
    resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower():
        raise RuntimeError(
            f"Expected a PDF from {url}, got Content-Type={content_type!r}. "
            "The media id is likely stale -- re-resolve it from the "
            "statute's finlex.fi/en/legislation/translations/... page."
        )
    return resp.content


def _extract_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [p.extract_text() or "" for p in reader.pages]


def _clean_lines(pages: list[str]) -> list[str]:
    lines: list[str] = []
    for page in pages:
        for raw in page.split("\n"):
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def parse_sections(pages: list[str]) -> list[dict]:
    """Parse cleaned page text into a list of section records.

    Each record: {chapter, chapter_no, section_no, section_title, body}
    """
    lines = _clean_lines(pages)
    sections: list[dict] = []
    current_chapter = ""
    current_chapter_no = ""
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m_chap = _CHAPTER_RE.match(line)
        if m_chap and i + 1 < n:
            current_chapter_no = m_chap.group(1)
            current_chapter = lines[i + 1]
            i += 2
            continue
        m_sec = _SECTION_RE.match(line)
        if m_sec and i + 1 < n:
            section_no = m_sec.group(1)
            section_title = lines[i + 1]
            i += 2
            body_lines = []
            while i < n and not _SECTION_RE.match(lines[i]) and not _CHAPTER_RE.match(lines[i]):
                body_lines.append(lines[i])
                i += 1
            body = " ".join(body_lines).strip()
            if body:
                sections.append(
                    {
                        "chapter": f"Chapter {current_chapter_no}: {current_chapter}"
                        if current_chapter
                        else "",
                        "section_no": section_no,
                        "section_title": section_title,
                        "body": body,
                    }
                )
            continue
        i += 1
    return sections


def ingest(source: FinlexStatuteSource) -> list[Chunk]:
    if source.local_pdf_path:
        with open(source.local_pdf_path, "rb") as f:
            pdf_bytes = f.read()
    elif source.media_id:
        pdf_bytes = fetch_pdf_bytes(source.media_id)
    else:
        raise ValueError("FinlexStatuteSource needs local_pdf_path or media_id")

    pages = _extract_pages(pdf_bytes)
    sections = parse_sections(pages)

    chunks: list[Chunk] = []
    today = utc_today()
    for sec in sections:
        chunk_id = f"{source.document_id}-s{sec['section_no']}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=source.document_id,
                document_title=source.document_title,
                text=f"Section {sec['section_no']} ({sec['section_title']}). {sec['body']}",
                source_url=source.finlex_page_url,
                licence=source.licence,
                retrieved_at=today,
                chapter=sec["chapter"],
                section_number=sec["section_no"],
                section_title=sec["section_title"],
            )
        )
    return chunks
