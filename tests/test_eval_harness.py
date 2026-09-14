"""Integration tests for the eval harness itself: golden.yaml has at
least 40 cases including genuine refusals, thresholds.yaml parses, and
the recorded fixtures are in sync with the golden set (every golden case
has a fixture, computed against the real, committed corpus -- no network
or LLM calls, since fixtures are pre-recorded).
"""

from __future__ import annotations

from eval.metrics import compute_metrics
from eval.run_eval import evaluate, load_fixtures, load_golden, load_thresholds


def test_golden_set_has_at_least_40_cases():
    golden = load_golden()
    assert len(golden) >= 40


def test_golden_set_has_both_answer_and_refuse_cases():
    golden = load_golden()
    behaviors = {c["expected_behavior"] for c in golden}
    assert behaviors == {"answer", "refuse"}
    refuse_cases = [c for c in golden if c["expected_behavior"] == "refuse"]
    assert len(refuse_cases) >= 5, "need a meaningful number of genuinely-unanswerable cases"


def test_golden_ids_are_unique():
    golden = load_golden()
    ids = [c["id"] for c in golden]
    assert len(ids) == len(set(ids))


def test_refuse_cases_have_no_expected_chunk_ids():
    golden = load_golden()
    for case in golden:
        if case["expected_behavior"] == "refuse":
            assert case["expected_chunk_ids"] == []


def test_thresholds_file_has_all_three_gated_metrics():
    thresholds = load_thresholds()
    for key in ("groundedness", "citation_precision", "refusal_correctness"):
        assert key in thresholds
        assert 0.0 <= thresholds[key] <= 1.0


def test_every_golden_case_has_a_recorded_fixture():
    golden = load_golden()
    fixtures = load_fixtures()
    missing = [c["id"] for c in golden if c["id"] not in fixtures]
    assert missing == [], f"missing fixtures for: {missing}"


def test_eval_against_committed_fixtures_meets_committed_thresholds():
    """This is the same computation eval/run_eval.py performs -- run here
    as a unit test so CI fails fast (with a clear pytest traceback) if
    someone edits golden.yaml/fixtures/thresholds out of sync, in
    addition to the standalone `python -m eval.run_eval` gate."""
    golden = load_golden()
    fixtures = load_fixtures()
    thresholds = load_thresholds()
    results = evaluate(golden, fixtures)
    report = compute_metrics(results)

    assert report.groundedness >= thresholds["groundedness"]
    assert report.citation_precision >= thresholds["citation_precision"]
    assert report.refusal_correctness >= thresholds["refusal_correctness"]
