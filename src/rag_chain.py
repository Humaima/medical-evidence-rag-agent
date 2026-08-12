"""
Medical Evidence RAG chain.

Pipeline:
  question
    -> FAISS retriever (top-k similar chunks + raw distances)
    -> prompt assembled with retrieved evidence, each chunk tagged with its doc_id
    -> LLM called with .with_structured_output(RAGAnswer) so the response is a
       validated Pydantic object, not free text we have to parse with regex
    -> confidence.decide() combines retrieval score + LLM self-confidence
    -> FinalResponse (answer + citations, or a refusal)
"""

from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # local embeddings -- no API key needed
from langchain_groq import ChatGroq             # chat/generation model
from langchain_core.documents import Document

from src.schemas import RAGAnswer
from src.confidence import compute_retrieval_stats, decide, RetrievalStats
from src.schemas import FinalResponse

load_dotenv()

# Must match the model used to build the index in ingest.py.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a clinical evidence assistant. You answer medical questions
ONLY using the evidence excerpts provided below. You must not use outside knowledge,
and you must not guess.

Rules:
1. Every factual claim in your answer must be traceable to one of the evidence excerpts.
2. Cite excerpts using their doc_id. Never invent a doc_id.
3. If the excerpts do not contain enough information to confidently answer the
   question, set is_answerable_from_evidence to false and give a low self_confidence,
   rather than filling gaps with general medical knowledge.
4. self_confidence should reflect how directly and completely the evidence supports
   your specific answer -- not how confident you are in general medical facts.
5. This tool is a research/decision-support demo. It does not replace professional
   medical judgment.
"""

USER_PROMPT_TEMPLATE = """Evidence excerpts:
{context}

Question: {question}

Respond with a structured answer following the required schema."""


def format_context(scored_docs: list[tuple[Document, float]]) -> str:
    blocks = []
    for doc, _dist in scored_docs:
        meta = doc.metadata
        blocks.append(
            f"[doc_id: {meta['doc_id']} | title: {meta['title']} | pmid: {meta.get('pmid', 'N/A')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


class MedicalRAGAgent:
    def __init__(
        self,
        index_path: str = "index/faiss_medical",
        # Groq's model lineup changes frequently and old names get decommissioned
        # outright (calls then fail with a 400 model_decommissioned error). If this
        # default ever breaks, list currently active models with:
        #   python -c "from groq import Groq; [print(m.id) for m in Groq().models.list().data if m.active]"
        # or check https://console.groq.com/docs/models.
        model_name: str = "llama-3.3-70b-versatile",
        top_k: int = 4,
    ):
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set.")

        index_dir = Path(index_path)
        if not index_dir.exists():
            raise FileNotFoundError(
                f"No FAISS index found at '{index_path}'. Run `python src/ingest.py` first."
            )

        # Embeddings: local sentence-transformers model, no API key/credits needed.
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
        self.top_k = top_k

        # Generation: routed to Groq (fast Llama 3.x inference).
        llm = ChatGroq(model=model_name, temperature=0)
        self.structured_llm = llm.with_structured_output(RAGAnswer)

    def retrieve(self, question: str) -> tuple[list[tuple[Document, float]], RetrievalStats]:
        scored_docs = self.vectorstore.similarity_search_with_score(question, k=self.top_k)
        stats = compute_retrieval_stats(scored_docs)
        return scored_docs, stats

    def answer(self, question: str) -> FinalResponse:
        scored_docs, retrieval_stats = self.retrieve(question)
        context = format_context(scored_docs)

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE.format(context=context, question=question)),
        ]

        rag_answer: RAGAnswer = self.structured_llm.invoke(messages) # type: ignore

        # Guard against the model citing a doc_id that wasn't actually retrieved --
        # a cheap but effective hallucination check on top of the schema itself.
        valid_ids = {doc.metadata["doc_id"] for doc, _ in scored_docs}
        rag_answer.citations = [c for c in rag_answer.citations if c.doc_id in valid_ids]

        return decide(rag_answer, retrieval_stats)
