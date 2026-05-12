"""
vectorstore.py — Embedding and FAISS vector-store management.

Provides three public functions:
  - build_vectorstore  — embed chunks and create an in-memory FAISS index
  - save_vectorstore   — persist the index to disk
  - load_vectorstore   — reload a persisted index from disk
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Embedding model constant — change here to swap models project-wide
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL = "models/gemini-embedding-001"


def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """
    Instantiate the Google Generative AI embeddings object.

    Kept as a private helper so that both build_vectorstore and
    load_vectorstore always use an identical embeddings configuration.

    Args:
        api_key: A valid Google AI Studio API key.

    Returns:
        A configured ``GoogleGenerativeAIEmbeddings`` instance.
    """
    return GoogleGenerativeAIEmbeddings(
        model=_EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def build_vectorstore(chunks: list[Document], api_key: str) -> FAISS:
    """
    Embed a list of Document chunks and build an in-memory FAISS index.

    Args:
        chunks:  List of ``langchain.schema.Document`` objects as returned
                 by ``chunker.chunk_text()``.
        api_key: A valid Google AI Studio API key used to call the
                 ``text-embedding-004`` model.

    Returns:
        A ``FAISS`` vector store loaded with all embedded chunks,
        ready for similarity search.

    Raises:
        ValueError: If *chunks* is empty.
        google.api_core.exceptions.GoogleAPIError: On any API-level error
            (invalid key, quota exceeded, etc.).
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    embeddings = _make_embeddings(api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vs: FAISS, path: str = "vector_index/") -> None:
    """
    Persist a FAISS vector store to disk.

    The index and document store are written to *path* using LangChain's
    ``save_local`` format (two files: ``index.faiss`` and ``index.pkl``).
    The directory is created automatically if it does not exist.

    Args:
        vs:   The ``FAISS`` vector store to persist.
        path: Destination directory path.  Defaults to ``"vector_index/"``.
    """
    vs.save_local(path)


def load_vectorstore(path: str = "vector_index/", api_key: str = "") -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.

    Re-initialises the same ``GoogleGenerativeAIEmbeddings`` model used
    during indexing so that query vectors are generated in the same
    embedding space.

    Args:
        path:    Directory where the index was saved.  Defaults to
                 ``"vector_index/"``.
        api_key: A valid Google AI Studio API key.

    Returns:
        A fully loaded ``FAISS`` vector store ready for similarity search.

    Raises:
        RuntimeError: If the index files are not found at *path*.
    """
    embeddings = _make_embeddings(api_key)
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
