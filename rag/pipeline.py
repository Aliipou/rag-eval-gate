"""LangGraph answering pipeline: retrieve -> generate -> verify -> finalize.

Four linear nodes, built as an actual `langgraph.graph.StateGraph` (not
just a plain function chain) so the pipeline has explicit, inspectable
state transitions:

    retrieve  -- VectorIndex.search(question)
    generate  -- Generator.generate(question, retrieved_chunks); every
                 claim it returns is paired with a chunk id
    verify    -- rag.verification.verify_claims(), pure, independent of
                 the generator; unsupported claims are dropped here,
                 never rewritten
    finalize  -- builds the answer text from surviving claims and the
                 TransparencyRecord; if nothing survives verification,
                 returns an explicit "not supported by sources" answer

`RETRIEVAL_SCORE_THRESHOLD` is a first refusal gate (don't even bother
generating if nothing relevant was retrieved); the second, stricter gate
is verification itself (if generation produced claims but none survive
verification, that's also a refusal). Both paths set `answered=False`
with a `refusal_reason` so the golden-set "should refuse" cases can be
scored precisely (see eval/metrics.py).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag.generation import Generator, ExtractiveGenerator, prompt_hash
from rag.retrieval import RetrievedChunk, TfidfVectorIndex, VectorIndex
from rag.transparency import TransparencyRecord
from rag.verification import Claim, VerificationResult, drop_unsupported, verify_claims

RETRIEVAL_K = 8
RETRIEVAL_SCORE_THRESHOLD = 0.15
NOT_SUPPORTED_MESSAGE = "This is not supported by the available sources."


class PipelineState(TypedDict, total=False):
    question: str
    retrieved: list[RetrievedChunk]
    claims: list[Claim]
    model_id: str
    raw_generation: str
    verification_results: list[VerificationResult]
    verified_claims: list[Claim]
    answer_text: str
    answered: bool
    refusal_reason: str | None
    transparency: TransparencyRecord


def _retrieve_node_factory(index: VectorIndex):
    def _node(state: PipelineState) -> PipelineState:
        retrieved = index.search(state["question"], k=RETRIEVAL_K)
        return {"retrieved": retrieved}

    return _node


def _generate_node_factory(generator: Generator):
    def _node(state: PipelineState) -> PipelineState:
        retrieved = state.get("retrieved", [])
        top_score = retrieved[0].score if retrieved else 0.0
        if top_score < RETRIEVAL_SCORE_THRESHOLD:
            return {
                "claims": [],
                "model_id": generator.model_id,
                "raw_generation": "",
                "refusal_reason": "no_relevant_chunks_retrieved",
            }
        output = generator.generate(state["question"], retrieved)
        return {
            "claims": output.claims,
            "model_id": output.model_id,
            "raw_generation": output.raw_text,
        }

    return _node


def _verify_node(state: PipelineState) -> PipelineState:
    chunks_by_id = {c.chunk_id: {"text": c.text} for c in state.get("retrieved", [])}
    results = verify_claims(state.get("claims", []), chunks_by_id)
    verified = drop_unsupported(state.get("claims", []), results)
    return {"verification_results": results, "verified_claims": verified}


def _finalize_node(state: PipelineState) -> PipelineState:
    verified = state.get("verified_claims", [])
    retrieved = state.get("retrieved", [])
    model_id = state.get("model_id", "unknown")

    if verified:
        answer_text = " ".join(f"{c.text} [{c.chunk_id}]" for c in verified)
        answered = True
        refusal_reason = None
    else:
        answer_text = NOT_SUPPORTED_MESSAGE
        answered = False
        refusal_reason = state.get("refusal_reason") or "no_claims_survived_verification"

    p_hash = prompt_hash(state["question"], retrieved, model_id)
    transparency = TransparencyRecord.build(
        model_id=model_id,
        prompt_hash=p_hash,
        question=state["question"],
        retrieved=retrieved,
        verification_results=state.get("verification_results", []),
        answered=answered,
        refusal_reason=refusal_reason,
    )
    return {
        "answer_text": answer_text,
        "answered": answered,
        "refusal_reason": refusal_reason,
        "transparency": transparency,
    }


def build_graph(index: VectorIndex, generator: Generator):
    graph = StateGraph(PipelineState)
    graph.add_node("retrieve", _retrieve_node_factory(index))
    graph.add_node("generate", _generate_node_factory(generator))
    graph.add_node("verify", _verify_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


class AnsweringPipeline:
    """Convenience wrapper around the compiled LangGraph graph."""

    def __init__(self, index: VectorIndex | None = None, generator: Generator | None = None):
        self.index = index or TfidfVectorIndex()
        self.generator = generator or ExtractiveGenerator()
        self._graph = build_graph(self.index, self.generator)

    def answer(self, question: str) -> PipelineState:
        return self._graph.invoke({"question": question})
