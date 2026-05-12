"""
qa_chain.py — Retrieval-Augmented Generation (RAG) Q&A chain.

Builds a RetrievalQA chain backed by the FAISS vector store created in
Sprint 1 and the Gemini 2.5 Flash LLM. All answers are grounded strictly
in the retrieved document context.
"""

from __future__ import annotations

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.vectorstore import load_vectorstore

# ---------------------------------------------------------------------------
# Prompt template — instructs the LLM to stay within the document context
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = """You are a helpful AI assistant.
You have been given the following extracted content from a PDF document.
Use this content to answer the user's question as helpfully as possible.

For broad questions like "summarize", "what is this about", or "key points":
- Give a full, detailed answer using ALL the context provided below
- Do not say you cannot find something if context is provided

For specific questions:
- Answer directly from the context
- If truly not present in the context, say: "This specific detail isn't in the
  sections I retrieved — try rephrasing or asking about a different part."

Context from document:
{context}

Question: {question}

Answer (be detailed and helpful):"""


def build_qa_chain(api_key: str) -> RetrievalQA:
    """
    Load the saved FAISS index and assemble a RetrievalQA chain.

    Steps performed internally:
    1. Loads the persisted FAISS vector store from ``vector_index/``.
    2. Wraps it as a retriever that returns the top-4 most relevant chunks.
    3. Initialises ``ChatGoogleGenerativeAI`` (gemini-2.5-flash, temp=0.2).
    4. Combines the retriever, LLM, and prompt into a ``RetrievalQA`` chain
       configured with ``chain_type="stuff"`` and
       ``return_source_documents=True``.

    Args:
        api_key: A valid Google AI Studio API key. Must match the key used
                 when the vector store was built.

    Returns:
        A fully configured ``RetrievalQA`` chain ready for inference.

    Raises:
        RuntimeError: If the FAISS index files are not found in
            ``vector_index/``.
    """
    # 1. Load the persisted vector store
    vs = load_vectorstore(path="vector_index/", api_key=api_key)

    # 2. Create retriever — top-6 chunks per query
    retriever = vs.as_retriever(search_kwargs={"k": 6})

    # 3. Initialise Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )

    # 4. Build the prompt
    prompt = PromptTemplate(
        template=_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # 5. Assemble the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def ask(chain: RetrievalQA, question: str) -> dict:
    """
    Run the RAG chain for a single question and return a structured result.

    The LLM's answer is grounded exclusively in the document chunks
    retrieved by the FAISS retriever. If no relevant content is found, the
    chain returns the fallback phrase defined in the prompt template.

    Args:
        chain:    A ``RetrievalQA`` chain as returned by ``build_qa_chain()``.
        question: The natural-language question to answer.

    Returns:
        A dict with two keys:

        ``"answer"`` — the LLM's response string (stripped of leading /
        trailing whitespace).

        ``"sources"`` — a deduplicated list of strings, each being the first
        150 characters of a retrieved source chunk's ``page_content``.
    """
    result = chain.invoke({"query": question})

    # Extract the answer text — key varies slightly across LangChain versions
    answer: str = result.get("result", result.get("answer", "")).strip()

    # Build deduplicated source snippets
    source_docs = result.get("source_documents", [])
    seen: set[str] = set()
    sources: list[str] = []
    for doc in source_docs:
        snippet = doc.page_content[:150].strip()
        if snippet not in seen:
            seen.add(snippet)
            sources.append(snippet)

    return {"answer": answer, "sources": sources}


def ask_with_memory(chain: RetrievalQA, question: str, chat_history: list) -> dict:
    """
    Run the RAG chain with full conversation history for context memory.

    Formats recent chat history and injects it into the question prompt so
    that the LLM can reference earlier parts of the conversation when
    answering follow-up questions.

    Args:
        chain:        A ``RetrievalQA`` chain as returned by ``build_qa_chain()``.
        question:     The user's current natural-language question.
        chat_history: List of dicts ``[{"role": "user"|"assistant", "content": str}]``
                      representing the full conversation so far.

    Returns:
        A dict with two keys:

        ``"answer"`` — the LLM's response string.

        ``"sources"`` — a deduplicated list of source-chunk preview strings.
    """
    # Format history as a readable string for context (last 6 messages = 3 turns)
    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Inject history into the question prompt
    augmented_question = f"""Previous conversation:
{history_text}
Current question: {question}

Answer the current question. If it refers to something in the conversation above, use that context."""

    result = chain.invoke({"query": augmented_question})

    answer: str = result.get("result", result.get("answer", "")).strip()

    source_docs = result.get("source_documents", [])
    seen: set[str] = set()
    sources: list[str] = []
    for doc in source_docs:
        snippet = doc.page_content[:150].strip()
        if snippet not in seen:
            seen.add(snippet)
            sources.append(snippet)

    return {"answer": answer, "sources": sources}
