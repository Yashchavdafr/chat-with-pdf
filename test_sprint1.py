"""
test_sprint1.py — End-to-end verification for the Sprint 1 ingestion pipeline.

Run with:
    python test_sprint1.py

Expected output (values vary by document):
    [LOADER]  Extracted 4321 characters
    [CHUNKER] 12 chunks created
    [VS]      Vector store built and saved to vector_index/
    [VS]      Vector store reloaded from disk
    [SEARCH]  Top result (first 200 chars):
              <text excerpt>
    SPRINT 1 COMPLETE ✓
"""

import os
import sys

# Ensure UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# ── Load API key from .env ────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print(
        "ERROR: GOOGLE_API_KEY not found.\n"
        "Create a .env file with:  GOOGLE_API_KEY=your_key_here"
    )
    sys.exit(1)

# ── Import pipeline modules ───────────────────────────────────────────────────
from app.loader import extract_text
from app.chunker import chunk_text
from app.vectorstore import build_vectorstore, save_vectorstore, load_vectorstore

PDF_PATH = "data/sample.pdf"
INDEX_PATH = "vector_index/"

def main() -> None:
    """Run the full Sprint 1 ingestion pipeline and print a summary."""

    # 1. Extract text from PDF
    print("\n[LOADER]  Extracting text from PDF …")
    text = extract_text(PDF_PATH)
    assert text, "extract_text() returned an empty string — check the PDF."
    print(f"[LOADER]  Extracted {len(text):,} characters")

    # 2. Chunk the text
    print("\n[CHUNKER] Splitting text into chunks …")
    chunks = chunk_text(text)
    assert len(chunks) > 0, "chunk_text() returned 0 chunks."
    print(f"[CHUNKER] {len(chunks)} chunks created")

    # 3. Build the vector store
    print("\n[VS]      Building FAISS vector store (this calls the Embeddings API) …")
    vs = build_vectorstore(chunks, api_key)
    print(f"[VS]      Vector store built — {len(chunks)} vectors indexed")

    # 4. Save to disk
    print(f"\n[VS]      Saving vector store to {INDEX_PATH} …")
    save_vectorstore(vs, INDEX_PATH)
    print(f"[VS]      Vector store saved to {INDEX_PATH}")

    # 5. Reload from disk
    print(f"\n[VS]      Reloading vector store from {INDEX_PATH} …")
    vs2 = load_vectorstore(INDEX_PATH, api_key)
    print(f"[VS]      Vector store reloaded from disk")

    # 6. Similarity search
    query = "What is this document about?"
    print(f"\n[SEARCH]  Running similarity search: '{query}'")
    results = vs2.similarity_search(query, k=3)
    assert results, "similarity_search() returned no results."
    top_content = results[0].page_content
    print(f"[SEARCH]  Top result (first 200 chars):\n          {top_content[:200]!r}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n  Total chunks  : {len(chunks)}")
    print(f"  Search results: {len(results)} returned")
    print("\nSPRINT 1 COMPLETE ✓")


if __name__ == "__main__":
    main()
