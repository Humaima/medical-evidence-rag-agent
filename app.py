"""
Interactive CLI demo for the Medical Evidence RAG Agent.

Usage:
    python app.py

Then type medical questions at the prompt. Type 'exit' to quit.

Try:
  - "How does metformin help with type 2 diabetes?"          (should answer, in-corpus)
  - "What is the recommended dosage of ibuprofen for a child?" (should refuse, out-of-corpus)
"""

from __future__ import annotations
import sys
from dotenv import load_dotenv

load_dotenv()

from src.rag_chain import MedicalRAGAgent  # noqa: E402


BANNER = """
==================================================================
 Medical Evidence RAG Agent  (demo)
 Retrieval-grounded answers with citation + confidence-based refusal
 NOT medical advice. For demonstration purposes only.
==================================================================
"""


def print_response(question: str, response) -> None:
    print(f"\nQ: {question}")
    print("-" * 70)
    if response.refused:
        print(f"[REFUSED] {response.answer}")
        print(f"  reason: {response.refusal_reason}")
    else:
        print(response.answer)
        if response.citations:
            print("\nCitations:")
            for c in response.citations:
                print(f"  - [{c.doc_id}] {c.title} (PMID: {c.pmid}) — \"{c.quoted_snippet}\"")
    print(
        f"\n[confidence: combined={response.confidence_score:.2f} "
        f"| retrieval={response.retrieval_score:.2f} "
        f"| llm_self={response.llm_self_confidence:.2f}]"
    )
    print("-" * 70)


def main() -> None:
    print(BANNER)
    try:
        agent = MedicalRAGAgent()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Setup error: {e}")
        sys.exit(1)

    while True:
        try:
            question = input("\nAsk a medical question ('exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = agent.answer(question)
        print_response(question, response)


if __name__ == "__main__":
    main()
