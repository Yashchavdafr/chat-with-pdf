"""
test_sprint2.py — End-to-end verification for the Sprint 2 RAG Q&A chain.

Run with:
    python test_sprint2.py

Expected output (content varies by document):
    Question: What is the main topic of this document?
    Answer:   <LLM answer>
    Sources used: 1
    ──────────────────────────────
    ...
    SPRINT 2 COMPLETE ✓
"""

import os
import sys

# Ensure UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# ── Load API key ──────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print(
        "ERROR: GOOGLE_API_KEY not found.\n"
        "Create a .env file with:  GOOGLE_API_KEY=your_key_here"
    )
    sys.exit(1)

# ── Import Sprint 2 module ────────────────────────────────────────────────────
from app.qa_chain import build_qa_chain, ask

# ── Test questions ────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarise the key points in 3 sentences.",
    "What conclusions does the document reach?",
]

DIVIDER = "─" * 50


def main() -> None:
    """Build the RAG chain, run 3 test questions, and verify all pass."""

    print("\nBuilding RAG Q&A chain …")
    chain = build_qa_chain(api_key)
    print("Chain ready.\n")
    print(DIVIDER)

    all_passed = True

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"Q{i}: {question}")
        result = ask(chain, question)

        answer = result["answer"]
        sources = result["sources"]

        print(f"Answer:      {answer}")
        print(f"Sources used: {len(sources)}")
        if sources:
            print(f"  └─ {sources[0][:120]} …")

        # Assertions
        if not answer:
            print(f"  [FAIL] Q{i}: answer is empty")
            all_passed = False
        if not sources:
            print(f"  [FAIL] Q{i}: no source chunks returned")
            all_passed = False

        print(DIVIDER)

    if all_passed:
        print("\nSPRINT 2 COMPLETE ✓")
    else:
        print("\nSome checks failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
