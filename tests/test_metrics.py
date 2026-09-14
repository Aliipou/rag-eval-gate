"""Tests for eval/metrics.py -- the three gated metrics plus the
informational retrieval_hit_rate."""

from __future__ import annotations

from eval.metrics import CaseResult, compute_metrics


def _case(
    case_id="q1",
    expected_behavior="answer",
    expected_chunk_ids=("c1",),
    retrieved_chunk_ids=("c1", "c2"),
    answered=True,
    claims=None,
    verification=None,
):
    return CaseResult(
        case_id=case_id,
        question="q?",
        expected_behavior=expected_behavior,
        expected_chunk_ids=list(expected_chunk_ids),
        retrieved_chunk_ids=list(retrieved_chunk_ids),
        answered=answered,
        claims=claims or [],
        verification=verification or [],
    )


def test_all_correct_gives_perfect_scores():
    case = _case(
        claims=[{"text": "Everyone is equal.", "chunk_id": "c1"}],
        verification=[{"claim_text": "Everyone is equal.", "chunk_id": "c1", "supported": True}],
    )
    report = compute_metrics([case])
    assert report.groundedness == 1.0
    assert report.citation_precision == 1.0
    assert report.refusal_correctness == 1.0
    assert report.retrieval_hit_rate == 1.0


def test_claim_citing_non_retrieved_chunk_hurts_groundedness():
    case = _case(
        retrieved_chunk_ids=("c1",),
        claims=[{"text": "Some claim.", "chunk_id": "c99"}],  # never retrieved
        verification=[{"claim_text": "Some claim.", "chunk_id": "c99", "supported": True}],
    )
    report = compute_metrics([case])
    assert report.groundedness == 0.0
    # citation_precision only checks verification.supported, unaffected here
    assert report.citation_precision == 1.0


def test_unsupported_claim_hurts_citation_precision_only():
    case = _case(
        claims=[{"text": "Fabricated claim.", "chunk_id": "c1"}],
        verification=[{"claim_text": "Fabricated claim.", "chunk_id": "c1", "supported": False}],
    )
    report = compute_metrics([case])
    assert report.groundedness == 1.0  # chunk_id WAS retrieved
    assert report.citation_precision == 0.0  # but not actually supported


def test_refusal_correctness_counts_mismatches():
    correct = _case(case_id="a", expected_behavior="answer", answered=True)
    wrong = _case(case_id="b", expected_behavior="refuse", answered=True)
    report = compute_metrics([correct, wrong])
    assert report.refusal_correctness == 0.5


def test_correct_refusal_is_not_penalised_for_zero_claims():
    case = _case(expected_behavior="refuse", expected_chunk_ids=(), answered=False, claims=[], verification=[])
    report = compute_metrics([case])
    assert report.refusal_correctness == 1.0
    # No claims at all -> groundedness/citation_precision default to 1.0 (vacuously true)
    assert report.groundedness == 1.0
    assert report.citation_precision == 1.0
    # retrieve_hit_rate only counts "answer" cases -> vacuously 1.0 here
    assert report.retrieval_hit_rate == 1.0


def test_retrieval_hit_rate_only_considers_answer_cases():
    hit = _case(case_id="a", expected_behavior="answer", expected_chunk_ids=("c1",), retrieved_chunk_ids=("c1", "c2"))
    miss = _case(case_id="b", expected_behavior="answer", expected_chunk_ids=("c9",), retrieved_chunk_ids=("c1", "c2"))
    refuse = _case(case_id="c", expected_behavior="refuse", expected_chunk_ids=(), retrieved_chunk_ids=(), answered=False)
    report = compute_metrics([hit, miss, refuse])
    assert report.retrieval_hit_rate == 0.5  # 1 of 2 "answer" cases hit; refuse case excluded


def test_empty_results_list_is_vacuously_perfect():
    report = compute_metrics([])
    assert report.groundedness == 1.0
    assert report.citation_precision == 1.0
    assert report.refusal_correctness == 1.0
    assert report.retrieval_hit_rate == 1.0
    assert report.n_cases == 0


def test_micro_average_weights_by_claim_count_not_case_count():
    # One case with 1 supported claim, one case with 3 unsupported claims.
    # Micro-average should be 1/4, not the per-case average of (1.0 + 0.0)/2 = 0.5.
    good = _case(
        case_id="a",
        claims=[{"text": "t1", "chunk_id": "c1"}],
        verification=[{"claim_text": "t1", "chunk_id": "c1", "supported": True}],
    )
    bad = _case(
        case_id="b",
        claims=[{"text": f"t{i}", "chunk_id": "c1"} for i in range(3)],
        verification=[{"claim_text": f"t{i}", "chunk_id": "c1", "supported": False} for i in range(3)],
    )
    report = compute_metrics([good, bad])
    assert report.citation_precision == 0.25
