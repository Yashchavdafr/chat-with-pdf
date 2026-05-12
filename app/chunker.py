"""
chunker.py — Text splitting module.

Splits a raw document string into overlapping chunks that are
suitable for embedding and vector-store retrieval.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_text(text: str) -> list[Document]:
    """
    Split a document string into overlapping LangChain Document chunks.

    Uses RecursiveCharacterTextSplitter with a chunk size of 500
    characters and an overlap of 50 characters so that context is
    preserved at chunk boundaries.

    Args:
        text: The full document text to split, typically the output
              of ``loader.extract_text()``.

    Returns:
        A list of ``langchain.schema.Document`` objects, each carrying
        a ``page_content`` field with the chunk text and an empty
        ``metadata`` dict.  Returns an empty list when *text* is empty
        or consists solely of whitespace.

    Example::

        from app.loader import extract_text
        from app.chunker import chunk_text

        raw = extract_text("data/sample.pdf")
        chunks = chunk_text(raw)
        print(len(chunks))   # e.g. 42
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        # Prefer splitting on paragraph / sentence / word boundaries
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = splitter.create_documents([text])
    return chunks
