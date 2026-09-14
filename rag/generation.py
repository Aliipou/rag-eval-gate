"""Claim generation: turn (question, retrieved chunks) into cited claims.

Every claim a generator returns MUST be paired with the id of the chunk
it claims to be supported by -- `rag/verification.py` then independently
checks that pairing. Generators never get to mark their own homework.

Two generators are provided:

- `ExtractiveGenerator` (default, always available): no LLM call at all.
  It selects sentences directly out of the retrieved chunks whose
  vocabulary overlaps the question, and cites each selected sentence
  with the chunk it was copied from. This is an honest stand-in for a
  real generative LLM, not a claim that it produces LLM-quality prose --
  it exists so the whole pipeline (retrieve -> generate -> verify ->
  transparency record) runs deterministically with zero API cost, which
  is what the recorded-fixture CI gate needs.
- `AnthropicGenerator` (live, optional): calls the Anthropic Messages API
  and asks the model to return claims as JSON, each tagged with a chunk
  id. This path is NOT exercised anywhere in this sandbox -- there is no
  ANTHROPIC_API_KEY available here. It exists for `--record` /
  the nightly live workflow once a real key is configured. See README.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

from rag.lexical import significant_tokens as _tokens
from rag.retrieval import RetrievedChunk
from rag.verification import Claim

EXTRACTIVE_MODEL_ID = "extractive-lexical-v1"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class GenerationOutput:
    claims: list[Claim]
    model_id: str
    raw_text: str


class Generator(Protocol):
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationOutput: ...


def prompt_hash(question: str, chunks: list[RetrievedChunk], model_id: str) -> str:
    """Stable hash of exactly what was sent to the generator.

    Included in every transparency record so a response can be tied back
    to the precise prompt (question + retrieved chunk ids/text + model)
    that produced it, without having to store the full prompt text
    everywhere.
    """
    payload = json.dumps(
        {
            "model_id": model_id,
            "question": question,
            "chunk_ids": [c.chunk_id for c in chunks],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class ExtractiveGenerator:
    """Deterministic, API-free generator used by default and in all tests/CI."""

    model_id = EXTRACTIVE_MODEL_ID

    def __init__(
        self,
        max_claims: int = 4,
        min_overlap: int = 2,
        high_confidence_chunk_score: float = 0.3,
    ):
        self.max_claims = max_claims
        self.min_overlap = min_overlap
        # A sentence from a chunk retrieval was very confident about (high
        # cosine similarity to the whole question) only needs to share one
        # significant word with the question, not `min_overlap`. This
        # avoids two failure modes seen while tuning against the golden
        # set: (a) marginal chunks matching on one generic shared word
        # (min_overlap=1 alone let "Finland"-only overlaps through on
        # off-topic questions), and (b) a strongly-relevant top chunk
        # producing zero claims because none of its sentences happens to
        # repeat >=2 of the question's exact words (min_overlap=2 alone
        # rejected clearly-relevant chunks like "Is property protected...?"
        # -> Section 15, whose only shared content word is "property").
        self.high_confidence_chunk_score = high_confidence_chunk_score

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationOutput:
        q_tokens = _tokens(question)
        candidates: list[tuple[float, str, str]] = []  # (score, sentence, chunk_id)

        for chunk in chunks:
            effective_min_overlap = (
                1 if chunk.score >= self.high_confidence_chunk_score else self.min_overlap
            )
            for sentence in _split_sentences(chunk.text):
                s_tokens = _tokens(sentence)
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens)
                if overlap < effective_min_overlap:
                    continue
                # Weight by retrieval score too, so higher-ranked chunks win ties.
                score = overlap + chunk.score
                candidates.append((score, sentence, chunk.chunk_id))

        candidates.sort(key=lambda t: t[0], reverse=True)
        seen_sentences: set[str] = set()
        claims: list[Claim] = []
        for _, sentence, chunk_id in candidates:
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            claims.append(Claim(text=sentence, chunk_id=chunk_id))
            if len(claims) >= self.max_claims:
                break

        raw_text = " ".join(c.text for c in claims)
        return GenerationOutput(claims=claims, model_id=self.model_id, raw_text=raw_text)


class AnthropicGenerator:
    """Live generator via the Anthropic Messages API. NOT exercised in this
    sandbox (no ANTHROPIC_API_KEY available) -- used only when a real key
    is configured, e.g. by `eval/run_eval.py --record` or the nightly
    live-eval GitHub Actions workflow.
    """

    def __init__(self, model_id: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model_id = model_id
        self._api_key = api_key

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationOutput:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed; run `pip install anthropic`"
            ) from e

        client = anthropic.Anthropic(api_key=self._api_key)  # reads ANTHROPIC_API_KEY if None
        context = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
        system = (
            "You answer questions using ONLY the numbered source chunks given below. "
            "Respond with a JSON array of claims. Each claim is an object with "
            '"text" (a single factual sentence) and "chunk_id" (the id of the '
            "chunk that supports it, exactly as given). Every sentence you assert "
            "must be traceable to one chunk id. If the chunks do not contain "
            "enough information to answer, return an empty JSON array []. "
            "Do not include any text outside the JSON array."
        )
        user = f"Source chunks:\n{context}\n\nQuestion: {question}"

        resp = client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = []

        claims = [
            Claim(text=item["text"], chunk_id=item["chunk_id"])
            for item in parsed
            if isinstance(item, dict) and "text" in item and "chunk_id" in item
        ]
        return GenerationOutput(claims=claims, model_id=self.model_id, raw_text=raw_text)
