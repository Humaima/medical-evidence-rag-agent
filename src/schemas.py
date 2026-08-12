"""
Pydantic schemas for the Medical Evidence RAG Agent.

These models force the LLM to return structured, machine-checkable output
instead of free-form text. This is what makes citation grounding and
confidence-based refusal possible: we are not asking the model to "please
be honest" -- we are asking it to fill in a typed schema, and then we
independently validate that schema before showing anything to the user.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """A single evidence citation backing a claim in the answer."""

    doc_id: str = Field(..., description="ID of the source document, e.g. 'doc003'")
    pmid: Optional[str] = Field(None, description="PubMed ID of the source, if available")
    title: str = Field(..., description="Title of the cited source")
    quoted_snippet: str = Field(
        ...,
        description="A short (<25 words) supporting snippet paraphrased from the source",
    )


class RAGAnswer(BaseModel):
    """
    Structured output the LLM must produce for every query.

    self_confidence is the model's own calibrated estimate of how well
    the retrieved evidence supports the answer. It is combined with the
    retriever's similarity scores (computed independently in Python, not
    by the LLM) to make the final refusal decision -- see confidence.py.
    """

    answer: str = Field(..., description="The answer to the user's medical question")
    citations: List[Citation] = Field(
        default_factory=list, description="Evidence citations supporting the answer"
    )
    self_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Model's own confidence (0-1) that the retrieved evidence "
            "sufficiently and directly supports the answer given"
        ),
    )
    reasoning_for_confidence: str = Field(
        ..., description="One or two sentences explaining the confidence score"
    )
    is_answerable_from_evidence: bool = Field(
        ...,
        description="False if the retrieved context does not contain enough information to answer",
    )

    @field_validator("citations")
    @classmethod
    def limit_citations(cls, v: List[Citation]) -> List[Citation]:
        return v[:5]


class FinalResponse(BaseModel):
    """
    What actually gets shown to the user, after Python-side confidence
    gating has been applied on top of the raw RAGAnswer.
    """

    answer: str
    citations: List[Citation]
    confidence_score: float
    refused: bool
    refusal_reason: Optional[str] = None
    retrieval_score: float
    llm_self_confidence: float
