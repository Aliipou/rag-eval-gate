"""Transparency record: the audit trail attached to every answer.

Modelled loosely on the kind of documentation the EU AI Act's
transparency provisions (Article 13/Article 50 territory -- logging,
traceability, explainability for the end user) point at. This project
does NOT claim to be EU AI Act compliant anywhere -- see README -- it
only borrows the idea that an AI system's output should be traceable:
which model produced it, what was retrieved, and which claims were
actually checked against source text and which were dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rag.retrieval import RetrievedChunk
from rag.verification import VerificationResult


@dataclass
class TransparencyRecord:
    model_id: str
    prompt_hash: str
    question: str
    retrieved_chunks: list[dict]  # [{chunk_id, document_id, score}]
    verification: list[dict]  # [{claim_text, chunk_id, supported, reason, coverage}]
    answered: bool
    refusal_reason: str | None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    @classmethod
    def build(
        cls,
        *,
        model_id: str,
        prompt_hash: str,
        question: str,
        retrieved: list[RetrievedChunk],
        verification_results: list[VerificationResult],
        answered: bool,
        refusal_reason: str | None,
    ) -> "TransparencyRecord":
        return cls(
            model_id=model_id,
            prompt_hash=prompt_hash,
            question=question,
            retrieved_chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "similarity_score": c.score,
                }
                for c in retrieved
            ],
            verification=[
                {
                    "claim_text": r.claim_text,
                    "chunk_id": r.chunk_id,
                    "supported": r.supported,
                    "reason": r.reason,
                    "coverage": r.coverage,
                }
                for r in verification_results
            ],
            answered=answered,
            refusal_reason=refusal_reason,
        )

    def to_json(self) -> dict:
        return {
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "question": self.question,
            "retrieved_chunks": self.retrieved_chunks,
            "verification": self.verification,
            "answered": self.answered,
            "refusal_reason": self.refusal_reason,
            "timestamp_utc": self.timestamp_utc,
        }

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_json(), indent=indent, ensure_ascii=False)
