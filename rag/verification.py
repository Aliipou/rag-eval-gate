"""Pure claim verification.

This is the safety-critical function in the whole project: the
generation step is only allowed to emit a claim paired with the chunk id
it says supports it, and this function is the sole judge of whether that
citation actually holds up. Unsupported claims get dropped by the
pipeline (`rag/pipeline.py`), never rewritten.

Design choice, stated plainly: there is no LLM/NLI model available in
this sandbox to do real entailment checking ("does this chunk logically
entail this claim?"). Instead this uses a deterministic *lexical
grounding* heuristic -- a claim is "supported" by a chunk when a high
enough fraction of the claim's significant (non-stopword) words actually
appear in that chunk's text. This is intentionally conservative and
occasionally too strict (a true paraphrase can fail it) or too loose (it
does not catch a claim that inverts a chunk's meaning while reusing its
vocabulary, e.g. negation). Both failure modes are called out in the
README and covered by test cases below/`tests/test_verification.py` so
the limitation is demonstrated, not hidden.

Every function here is pure: no I/O, no randomness, no network calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.lexical import significant_tokens as _significant_tokens_impl

DEFAULT_COVERAGE_THRESHOLD = 0.6
MIN_SIGNIFICANT_TOKENS = 1


@dataclass(frozen=True)
class Claim:
    text: str
    chunk_id: str


@dataclass(frozen=True)
class VerificationResult:
    claim_text: str
    chunk_id: str
    supported: bool
    reason: str
    coverage: float


def _significant_tokens(text: str) -> set[str]:
    return _significant_tokens_impl(text)


def verify_claim(
    claim: Claim,
    chunks_by_id: dict[str, dict],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> VerificationResult:
    """Verify a single claim against the chunk it cites.

    Pure function: same inputs always produce the same output.
    """
    chunk = chunks_by_id.get(claim.chunk_id)
    if chunk is None:
        return VerificationResult(
            claim_text=claim.text,
            chunk_id=claim.chunk_id,
            supported=False,
            reason="unknown_chunk_id",
            coverage=0.0,
        )

    claim_tokens = _significant_tokens(claim.text)
    if len(claim_tokens) < MIN_SIGNIFICANT_TOKENS:
        return VerificationResult(
            claim_text=claim.text,
            chunk_id=claim.chunk_id,
            supported=False,
            reason="claim_has_no_content_words",
            coverage=0.0,
        )

    chunk_tokens = _significant_tokens(chunk["text"])
    overlap = claim_tokens & chunk_tokens
    coverage = len(overlap) / len(claim_tokens)

    if coverage >= threshold:
        return VerificationResult(
            claim_text=claim.text,
            chunk_id=claim.chunk_id,
            supported=True,
            reason="lexical_coverage_ok",
            coverage=round(coverage, 4),
        )
    return VerificationResult(
        claim_text=claim.text,
        chunk_id=claim.chunk_id,
        supported=False,
        reason="lexical_coverage_below_threshold",
        coverage=round(coverage, 4),
    )


def verify_claims(
    claims: list[Claim],
    chunks_by_id: dict[str, dict],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> list[VerificationResult]:
    return [verify_claim(c, chunks_by_id, threshold=threshold) for c in claims]


def drop_unsupported(
    claims: list[Claim],
    results: list[VerificationResult],
) -> list[Claim]:
    """Filter claims down to only the ones verification supported.

    Unsupported claims are dropped, never rewritten -- rewriting would
    mean generating new, unverified text to paper over a failed
    citation, which defeats the point of verification.
    """
    supported_ids = {
        (r.claim_text, r.chunk_id) for r in results if r.supported
    }
    return [c for c in claims if (c.text, c.chunk_id) in supported_ids]
