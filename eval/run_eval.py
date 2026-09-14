"""The eval gate.

Default mode (what CI runs): loads eval/golden.yaml and the recorded
fixtures in eval/fixtures/recorded_responses.json, re-runs the PURE
verification function against the live corpus text, computes
groundedness / citation_precision / refusal_correctness, compares each
against eval/thresholds.yaml, and exits non-zero if any metric falls
below its threshold. No network calls, no LLM calls, no randomness --
same input always gives the same output.

    python -m eval.run_eval

`--record` mode re-runs retrieval + generation for every golden case
(by default via the deterministic ExtractiveGenerator; pass
`--generator anthropic` to use the live Anthropic API, which needs
ANTHROPIC_API_KEY and is NOT exercised in this sandbox) and overwrites
eval/fixtures/recorded_responses.json. Run this, inspect the diff, and
commit it whenever the corpus, retrieval, or generator changes.

    python -m eval.run_eval --record
    python -m eval.run_eval --record --generator anthropic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from eval.metrics import CaseResult, compute_metrics
from rag.generation import AnthropicGenerator, ExtractiveGenerator
from rag.retrieval import TfidfVectorIndex, load_chunks
from rag.verification import Claim, drop_unsupported, verify_claims

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "eval" / "golden.yaml"
THRESHOLDS_PATH = ROOT / "eval" / "thresholds.yaml"
FIXTURES_PATH = ROOT / "eval" / "fixtures" / "recorded_responses.json"
RETRIEVAL_K = 8
RETRIEVAL_SCORE_THRESHOLD = 0.15  # kept in sync with rag/pipeline.py


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fixtures(path: Path = FIXTURES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def record_fixtures(
    golden: list[dict],
    generator_name: str = "extractive",
    out_path: Path = FIXTURES_PATH,
) -> dict:
    index = TfidfVectorIndex()
    if generator_name == "extractive":
        generator = ExtractiveGenerator()
    elif generator_name == "anthropic":
        generator = AnthropicGenerator()
    else:
        raise ValueError(f"Unknown generator: {generator_name}")

    fixtures = {}
    for case in golden:
        question = case["question"]
        retrieved = index.search(question, k=RETRIEVAL_K)
        top_score = retrieved[0].score if retrieved else 0.0
        if top_score < RETRIEVAL_SCORE_THRESHOLD:
            claims = []
            model_id = generator.model_id
        else:
            output = generator.generate(question, retrieved)
            claims = [{"text": c.text, "chunk_id": c.chunk_id} for c in output.claims]
            model_id = output.model_id

        fixtures[case["id"]] = {
            "question": question,
            "model_id": model_id,
            "retrieved": [
                {"chunk_id": c.chunk_id, "document_id": c.document_id, "score": c.score}
                for c in retrieved
            ],
            "claims": claims,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return fixtures


def evaluate(golden: list[dict], fixtures: dict) -> list[CaseResult]:
    chunks_by_id = {c["chunk_id"]: c for c in load_chunks()}
    results = []
    for case in golden:
        fixture = fixtures.get(case["id"])
        if fixture is None:
            raise KeyError(
                f"No recorded fixture for case {case['id']!r} -- run "
                "`python -m eval.run_eval --record` first."
            )
        claims = [Claim(text=c["text"], chunk_id=c["chunk_id"]) for c in fixture["claims"]]
        verification = verify_claims(claims, chunks_by_id)
        verified = drop_unsupported(claims, verification)

        results.append(
            CaseResult(
                case_id=case["id"],
                question=case["question"],
                expected_behavior=case["expected_behavior"],
                expected_chunk_ids=case.get("expected_chunk_ids", []),
                retrieved_chunk_ids=[r["chunk_id"] for r in fixture["retrieved"]],
                answered=bool(verified),
                claims=fixture["claims"],
                verification=[
                    {
                        "claim_text": v.claim_text,
                        "chunk_id": v.chunk_id,
                        "supported": v.supported,
                        "reason": v.reason,
                        "coverage": v.coverage,
                    }
                    for v in verification
                ],
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Refresh fixtures and exit")
    parser.add_argument(
        "--generator",
        default="extractive",
        choices=["extractive", "anthropic"],
        help="Generator to use with --record (default: extractive, no API key needed)",
    )
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--thresholds", default=str(THRESHOLDS_PATH))
    parser.add_argument("--fixtures", default=str(FIXTURES_PATH))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    args = parser.parse_args()

    golden = load_golden(Path(args.golden))

    if args.record:
        fixtures = record_fixtures(golden, generator_name=args.generator, out_path=Path(args.fixtures))
        print(f"Recorded {len(fixtures)} fixtures to {args.fixtures} (generator={args.generator})")
        return 0

    fixtures = load_fixtures(Path(args.fixtures))
    results = evaluate(golden, fixtures)
    report = compute_metrics(results)
    thresholds = load_thresholds(Path(args.thresholds))

    gated = {
        "groundedness": report.groundedness,
        "citation_precision": report.citation_precision,
        "refusal_correctness": report.refusal_correctness,
    }

    failures = []
    for name, value in gated.items():
        threshold = thresholds.get(name)
        if threshold is None:
            continue
        if value < threshold:
            failures.append((name, value, threshold))

    if args.json:
        print(json.dumps({**report.to_json(), "thresholds": thresholds, "failures": failures}, indent=2))
    else:
        print("=== Eval gate report ===")
        print(f"cases: {report.n_cases}   claims: {report.n_claims}")
        for name, value in gated.items():
            threshold = thresholds.get(name)
            status = "PASS"
            if threshold is not None and value < threshold:
                status = "FAIL"
            print(f"  {name:20s} {value:.4f}   threshold={threshold}   [{status}]")
        print(f"  {'retrieval_hit_rate':20s} {report.retrieval_hit_rate:.4f}   (informational, not gated)")

        if failures:
            print("\nFailing cases (refusal-behavior mismatches):")
            for pc in report.per_case:
                if not pc["behavior_correct"]:
                    print(f"  [{pc['id']}] expected={pc['expected_behavior']} answered={pc['answered']} :: {pc['question']}")

    if failures:
        print(f"\nEVAL GATE FAILED: {len(failures)} metric(s) below threshold")
        for name, value, threshold in failures:
            print(f"  - {name}: {value:.4f} < {threshold}")
        return 1

    print("\nEVAL GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
