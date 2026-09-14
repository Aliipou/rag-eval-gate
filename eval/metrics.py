"""Metrics computed by the eval gate.

Three metrics, matching the project brief exactly:

- **groundedness**: of all claims a generator produced, what fraction
  cite a chunk id that was actually retrieved for that question? This
  is a *structural* check -- did the generator stay inside the context
  it was given, rather than citing a chunk id it invented or that
  belongs to a different question. (A real LLM can and does fabricate
  citations; the deterministic extractive generator used in this
  sandbox can't, by construction, which is itself a limitation worth
  being honest about -- see README.)
- **citation_precision**: of all claims, what fraction are actually
  supported by the *text* of the chunk they cite, per
  `rag.verification.verify_claim`'s lexical-grounding check? This is
  the *content* check: does the cited chunk really contain the claim.
- **refusal_correctness**: of all golden cases, what fraction did the
  system's answered/refused decision match `expected_behavior`?

All three are computed as simple fractions (micro-averaged across the
whole golden set, not averaged-of-per-case-averages) so one badly wrong
case can't hide behind many trivial ones.

`retrieval_hit_rate` is also reported (fraction of "answer" cases where
at least one `expected_chunk_id` was actually retrieved) as a diagnostic
-- it is NOT one of the three gated metrics and has no threshold, since
the brief specifies exactly three.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseResult:
    case_id: str
    question: str
    expected_behavior: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    answered: bool
    claims: list[dict]  # [{"text", "chunk_id"}]
    verification: list[dict]  # [{"claim_text", "chunk_id", "supported", ...}]


@dataclass
class MetricsReport:
    groundedness: float
    citation_precision: float
    refusal_correctness: float
    retrieval_hit_rate: float
    n_cases: int
    n_claims: int
    per_case: list[dict]

    def to_json(self) -> dict:
        return {
            "groundedness": self.groundedness,
            "citation_precision": self.citation_precision,
            "refusal_correctness": self.refusal_correctness,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "n_cases": self.n_cases,
            "n_claims": self.n_claims,
        }


def compute_metrics(results: list[CaseResult]) -> MetricsReport:
    total_claims = 0
    grounded_claims = 0  # claim's chunk_id was actually retrieved
    precise_claims = 0  # claim's chunk_id text actually supports it (verification.supported)

    refusal_correct = 0
    answer_cases = 0
    retrieval_hits = 0

    per_case = []

    for r in results:
        retrieved_set = set(r.retrieved_chunk_ids)
        verif_by_key = {
            (v["claim_text"], v["chunk_id"]): v["supported"] for v in r.verification
        }

        case_claims = len(r.claims)
        case_grounded = 0
        case_precise = 0
        for claim in r.claims:
            total_claims += 1
            if claim["chunk_id"] in retrieved_set:
                grounded_claims += 1
                case_grounded += 1
            if verif_by_key.get((claim["text"], claim["chunk_id"]), False):
                precise_claims += 1
                case_precise += 1

        expected_answer = r.expected_behavior == "answer"
        behavior_correct = r.answered == expected_answer
        refusal_correct += int(behavior_correct)

        if expected_answer:
            answer_cases += 1
            if retrieved_set & set(r.expected_chunk_ids):
                retrieval_hits += 1

        per_case.append(
            {
                "id": r.case_id,
                "question": r.question,
                "expected_behavior": r.expected_behavior,
                "answered": r.answered,
                "behavior_correct": behavior_correct,
                "n_claims": case_claims,
                "n_grounded": case_grounded,
                "n_precise": case_precise,
            }
        )

    groundedness = grounded_claims / total_claims if total_claims else 1.0
    citation_precision = precise_claims / total_claims if total_claims else 1.0
    refusal_correctness = refusal_correct / len(results) if results else 1.0
    retrieval_hit_rate = retrieval_hits / answer_cases if answer_cases else 1.0

    return MetricsReport(
        groundedness=round(groundedness, 4),
        citation_precision=round(citation_precision, 4),
        refusal_correctness=round(refusal_correctness, 4),
        retrieval_hit_rate=round(retrieval_hit_rate, 4),
        n_cases=len(results),
        n_claims=total_claims,
        per_case=per_case,
    )
