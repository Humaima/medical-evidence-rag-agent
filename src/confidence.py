"""
Confidence handling and refusal logic.

This is the "safety" layer of the project. Two independent confidence
signals are combined so that the system does not have to trust the LLM's
self-report alone (LLMs are frequently overconfident):

1. retrieval_score  -- similarity between the question and the best
   matching evidence chunk(s), computed by FAISS/embeddings. This is a
   purely numerical, model-independent signal: if nothing in the corpus
   is actually close to the question, this will be low no matter what
   the LLM claims.

2. llm_self_confidence -- the model's own calibrated 0-1 confidence,
   returned as part of the structured RAGAnswer, plus an explicit
   is_answerable_from_evidence boolean.

The final decision to answer or refuse requires BOTH signals to clear
their thresholds. This "AND" gate is the key design choice: a single
weak signal is enough to trigger refusal, which biases the system
toward caution -- appropriate for a clinical-decision-support demo.
"""

from __future__ import annotations
from dataclasses import dataclass

from src.schemas import RAGAnswer, FinalResponse, Citation


# --- Tunable thresholds -----------------------------------------------
# These are demo defaults; in a real deployment they'd be tuned against
# a labeled evaluation set of answerable vs. unanswerable questions.
RETRIEVAL_SCORE_THRESHOLD = 0.55   # normalized similarity, 0-1, higher = closer match
LLM_CONFIDENCE_THRESHOLD = 0.60    # model's own confidence, 0-1

REFUSAL_MESSAGE = (
    "I don't have sufficient evidence in the current corpus to answer this "
    "question reliably. Please consult a qualified clinician or provide "
    "additional trusted sources."
)


@dataclass
class RetrievalStats:
    """Similarity stats computed from the raw FAISS search, independent of the LLM."""

    top_score: float          # normalized similarity of best-matching chunk, 0-1
    mean_top_k_score: float   # normalized average similarity across retrieved chunks


def normalize_faiss_distance(distance: float) -> float:
    """
    FAISS's similarity_search_with_score returns an L2 distance (lower = closer)
    for the default index. We convert it to a 0-1 "similarity-like" score so it
    can be compared against a single intuitive threshold.

    This is a simple monotonic transform, not a calibrated probability --
    good enough for a demo threshold, not for production risk scoring.
    """
    return 1.0 / (1.0 + distance)


def compute_retrieval_stats(scored_docs: list[tuple]) -> RetrievalStats:
    """scored_docs: list of (Document, raw_l2_distance) as returned by FAISS."""
    if not scored_docs:
        return RetrievalStats(top_score=0.0, mean_top_k_score=0.0)

    similarities = [normalize_faiss_distance(dist) for _, dist in scored_docs]
    return RetrievalStats(
        top_score=max(similarities),
        mean_top_k_score=sum(similarities) / len(similarities),
    )


def decide(
    rag_answer: RAGAnswer,
    retrieval_stats: RetrievalStats,
    retrieval_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    llm_threshold: float = LLM_CONFIDENCE_THRESHOLD,
) -> FinalResponse:
    """
    Apply the AND-gate confidence check and produce the final, user-facing response.
    """
    combined_confidence = min(retrieval_stats.top_score, rag_answer.self_confidence)

    reasons = []
    if retrieval_stats.top_score < retrieval_threshold:
        reasons.append(
            f"retrieved evidence similarity ({retrieval_stats.top_score:.2f}) "
            f"is below threshold ({retrieval_threshold:.2f})"
        )
    if rag_answer.self_confidence < llm_threshold:
        reasons.append(
            f"model self-confidence ({rag_answer.self_confidence:.2f}) "
            f"is below threshold ({llm_threshold:.2f})"
        )
    if not rag_answer.is_answerable_from_evidence:
        reasons.append("model flagged the question as not answerable from the retrieved evidence")

    should_refuse = len(reasons) > 0

    if should_refuse:
        return FinalResponse(
            answer=REFUSAL_MESSAGE,
            citations=[],
            confidence_score=combined_confidence,
            refused=True,
            refusal_reason="; ".join(reasons),
            retrieval_score=retrieval_stats.top_score,
            llm_self_confidence=rag_answer.self_confidence,
        )

    return FinalResponse(
        answer=rag_answer.answer,
        citations=rag_answer.citations,
        confidence_score=combined_confidence,
        refused=False,
        refusal_reason=None,
        retrieval_score=retrieval_stats.top_score,
        llm_self_confidence=rag_answer.self_confidence,
    )
