"""Exhaustive, table-driven tests for the pure claim-verification function.

This is the safety-critical function in the project (see rag/verification.py
docstring), so it gets the most thorough test treatment: every branch of
`verify_claim` is covered by an explicit case below, plus property-style
checks (purity/determinism) and `drop_unsupported` filtering behaviour.
No I/O, no mocking needed -- the function is pure.
"""

from __future__ import annotations

import pytest

from rag.verification import (
    Claim,
    VerificationResult,
    drop_unsupported,
    verify_claim,
    verify_claims,
)

CHUNKS = {
    "c1": {"text": "Everyone is equal before the law. No one shall be treated differently without an acceptable reason."},
    "c2": {"text": "The property of everyone is safeguarded by the constitution."},
    "c3": {"text": "The Bank of Finland operates under the guarantee and care of Parliament."},
    "empty": {"text": ""},
}

# (description, claim, expected_supported, expected_reason)
TABLE = [
    (
        "exact sentence from its own chunk is supported",
        Claim(text="Everyone is equal before the law.", chunk_id="c1"),
        True,
        "lexical_coverage_ok",
    ),
    (
        "paraphrase with strong word overlap is supported",
        Claim(text="Everyone is equal before the law and cannot be treated differently.", chunk_id="c1"),
        True,
        "lexical_coverage_ok",
    ),
    (
        "claim citing the wrong chunk is not supported",
        Claim(text="Everyone is equal before the law.", chunk_id="c2"),
        False,
        "lexical_coverage_below_threshold",
    ),
    (
        "claim citing a chunk id that does not exist",
        Claim(text="Everyone is equal before the law.", chunk_id="does-not-exist"),
        False,
        "unknown_chunk_id",
    ),
    (
        "claim about unrelated content is not supported",
        Claim(text="The Bank of Finland operates under Parliament.", chunk_id="c1"),
        False,
        "lexical_coverage_below_threshold",
    ),
    (
        "claim against an empty chunk is not supported",
        Claim(text="Everyone is equal before the law.", chunk_id="empty"),
        False,
        "lexical_coverage_below_threshold",
    ),
    (
        "claim with only stopwords / no content words",
        Claim(text="It is the and of.", chunk_id="c1"),
        False,
        "claim_has_no_content_words",
    ),
    (
        "empty claim text",
        Claim(text="", chunk_id="c1"),
        False,
        "claim_has_no_content_words",
    ),
    (
        "case-insensitive matching",
        Claim(text="EVERYONE IS EQUAL BEFORE THE LAW.", chunk_id="c1"),
        True,
        "lexical_coverage_ok",
    ),
    (
        "stemmed plural matches singular in chunk",
        Claim(text="Properties are safeguarded.", chunk_id="c2"),
        True,
        "lexical_coverage_ok",
    ),
    (
        "single strong keyword match still below default threshold when claim is long",
        Claim(
            text="The property of everyone is regulated by an entirely separate unrelated ministry process.",
            chunk_id="c2",
        ),
        False,
        "lexical_coverage_below_threshold",
    ),
    (
        "claim correctly cites the bank of finland chunk",
        Claim(text="The Bank of Finland operates under the guarantee of Parliament.", chunk_id="c3"),
        True,
        "lexical_coverage_ok",
    ),
]


@pytest.mark.parametrize("description,claim,expected_supported,expected_reason", TABLE, ids=[t[0] for t in TABLE])
def test_verify_claim_table(description, claim, expected_supported, expected_reason):
    result = verify_claim(claim, CHUNKS)
    assert result.supported == expected_supported, description
    assert result.reason == expected_reason, description


def test_verify_claim_is_pure_and_deterministic():
    claim = Claim(text="Everyone is equal before the law.", chunk_id="c1")
    r1 = verify_claim(claim, CHUNKS)
    r2 = verify_claim(claim, CHUNKS)
    assert r1 == r2


def test_verify_claim_does_not_mutate_inputs():
    chunks_copy = {k: dict(v) for k, v in CHUNKS.items()}
    claim = Claim(text="Everyone is equal before the law.", chunk_id="c1")
    verify_claim(claim, chunks_copy)
    assert chunks_copy == CHUNKS


def test_threshold_is_configurable():
    # A weak partial-overlap claim: fails at the strict default threshold,
    # passes once the threshold is loosened.
    claim = Claim(text="Everyone before the law and property are both safeguarded topics.", chunk_id="c1")
    strict = verify_claim(claim, CHUNKS, threshold=0.9)
    loose = verify_claim(claim, CHUNKS, threshold=0.1)
    assert strict.supported is False
    assert loose.supported is True


def test_verify_claims_batches_in_order():
    claims = [
        Claim(text="Everyone is equal before the law.", chunk_id="c1"),
        Claim(text="Not real content.", chunk_id="does-not-exist"),
    ]
    results = verify_claims(claims, CHUNKS)
    assert [r.supported for r in results] == [True, False]
    assert [r.chunk_id for r in results] == ["c1", "does-not-exist"]


class TestDropUnsupported:
    def test_keeps_only_supported_claims(self):
        claims = [
            Claim(text="Everyone is equal before the law.", chunk_id="c1"),
            Claim(text="The Bank of Finland operates under Parliament.", chunk_id="c1"),
        ]
        results = verify_claims(claims, CHUNKS)
        survivors = drop_unsupported(claims, results)
        assert survivors == [claims[0]]

    def test_empty_input(self):
        assert drop_unsupported([], []) == []

    def test_all_supported_survive(self):
        claims = [Claim(text="Everyone is equal before the law.", chunk_id="c1")]
        results = verify_claims(claims, CHUNKS)
        assert drop_unsupported(claims, results) == claims

    def test_duplicate_claims_both_handled(self):
        claims = [
            Claim(text="Everyone is equal before the law.", chunk_id="c1"),
            Claim(text="Everyone is equal before the law.", chunk_id="c1"),
        ]
        results = verify_claims(claims, CHUNKS)
        assert drop_unsupported(claims, results) == claims
