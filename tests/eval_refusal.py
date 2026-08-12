"""
Small evaluation harness demonstrating the confidence-based refusal
mechanism on a handful of hand-labeled questions.

This is intentionally simple (no pytest dependency) so it can be run
directly and the output pasted into a README/portfolio screenshot.

Usage:
    python tests/eval_refusal.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from src.rag_chain import MedicalRAGAgent  # noqa: E402


# expected=True  -> question SHOULD be answerable from the sample corpus
# expected=False -> question SHOULD trigger refusal (out of corpus / unsupported)
TEST_CASES = [
    ("Why is metformin used as first-line therapy for type 2 diabetes?", True),
    ("How do statins reduce cardiovascular risk?", True),
    ("What is the current guidance on aspirin for primary prevention?", True),
    ("What is the recommended pediatric dosage of amoxicillin for otitis media?", False),
    ("Does turmeric supplementation cure autoimmune disease?", False),
    ("What is the average life expectancy on Mars colonies?", False),
]


def main() -> None:
    agent = MedicalRAGAgent()
    correct = 0

    print(f"{'Question':<70} {'Expected':<10} {'Got':<10} {'Result'}")
    print("-" * 110)

    for question, expected_answerable in TEST_CASES:
        response = agent.answer(question)
        got_answerable = not response.refused
        is_correct = got_answerable == expected_answerable
        correct += int(is_correct)

        print(
            f"{question[:68]:<70} "
            f"{'ANSWER' if expected_answerable else 'REFUSE':<10} "
            f"{'ANSWER' if got_answerable else 'REFUSE':<10} "
            f"{'OK' if is_correct else 'MISS'}"
        )

    print("-" * 110)
    print(f"Accuracy: {correct}/{len(TEST_CASES)} ({100 * correct / len(TEST_CASES):.0f}%)")


if __name__ == "__main__":
    main()
