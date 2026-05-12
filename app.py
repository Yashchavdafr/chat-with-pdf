"""
app.py — Chat with PDF · Sprint 3 Streamlit UI

Two-screen dark glassmorphism interface:
  Screen 1: Upload PDF → animated processing stages
  Screen 2: Chat with typewriter answers + source count + conversation memory

Run:
    streamlit run app.py

API key is read from:
  1. .env file (local dev via python-dotenv)
  2. Environment variable GOOGLE_API_KEY (Streamlit Cloud secrets)
"""

import os
import time
import tempfile
import pathlib

import streamlit as st
from dotenv import load_dotenv

# ── Sprint 1 + 2 backend ──────────────────────────────────────────────────────
from app.loader import extract_text
from app.chunker import chunk_text
from app.vectorstore import build_vectorstore, save_vectorstore
from app.qa_chain import build_qa_chain, ask, ask_with_memory

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── API Key: works locally (.env) AND on Streamlit Cloud (secrets) ────────────
load_dotenv()
api_key = (
    os.environ.get("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or st.secrets.get("GOOGLE_API_KEY", "")
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = pathlib.Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ── Session State Defaults ────────────────────────────────────────────────────
defaults = {
    "ready": False,
    "qa_chain": None,
    "chat_history": [],
    "filename": "",
    "thinking": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Animated Hero SVG ─────────────────────────────────────────────────────────
HERO_SVG = """
<div class="hero-icon">
  <svg width="120" height="120" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="60" cy="60" r="50" stroke="url(#grad1)" stroke-width="2" opacity="0.3"
            style="animation: spinRing 8s linear infinite; transform-origin: center;" />
    <circle cx="60" cy="60" r="30" fill="url(#grad2)" opacity="0.15"
            style="animation: orbPulse 3s ease-in-out infinite;" />
    <circle cx="60" cy="42" r="4" fill="#6366f1"
            style="animation: floatDot 2s ease-in-out infinite;" />
    <circle cx="48" cy="68" r="3" fill="#06b6d4"
            style="animation: floatDot 2.5s ease-in-out infinite;" />
    <circle cx="72" cy="68" r="3" fill="#818cf8"
            style="animation: floatDot 2.2s ease-in-out infinite;" />
    <rect x="46" y="50" width="28" height="24" rx="3" fill="none"
          stroke="#6366f1" stroke-width="1.5" opacity="0.7" />
    <line x1="52" y1="58" x2="68" y2="58" stroke="#06b6d4" stroke-width="1.5" opacity="0.5" />
    <line x1="52" y1="63" x2="64" y2="63" stroke="#06b6d4" stroke-width="1.5" opacity="0.4" />
    <line x1="52" y1="68" x2="60" y2="68" stroke="#06b6d4" stroke-width="1.5" opacity="0.3" />
    <defs>
      <linearGradient id="grad1" x1="0" y1="0" x2="120" y2="120">
        <stop offset="0%" stop-color="#6366f1" />
        <stop offset="100%" stop-color="#06b6d4" />
      </linearGradient>
      <radialGradient id="grad2" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#6366f1" />
        <stop offset="100%" stop-color="#06b6d4" />
      </radialGradient>
    </defs>
  </svg>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: process an uploaded PDF through the full Sprint 1+2 pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_pdf(uploaded_file) -> None:
    """
    Run the full ingestion + chain-build pipeline for an uploaded PDF.

    Stages (each reflected in the animated progress bar):
    1. Read PDF bytes from the upload widget and write to a temp file.
    2. extract_text()       — loader
    3. chunk_text()         — chunker
    4. build_vectorstore()  — embedding + FAISS
    5. save_vectorstore()   — persist to vector_index/
    6. build_qa_chain()     — assemble RAG chain
    7. Mark session as ready.

    Args:
        uploaded_file: Streamlit UploadedFile object from st.file_uploader().
    """
    stages = [
        (0.15, "📖 Reading your document..."),
        (0.35, "✂️  Splitting into chunks..."),
        (0.60, "🧠 Building knowledge index..."),
        (0.85, "⚡ Connecting to Gemini..."),
        (1.00, "✅ Ready to chat!"),
    ]

    progress_bar = st.progress(0)
    status_text  = st.empty()

    # ── Stage 1: Save upload to temp file ────────────────────────────────────
    status_text.markdown(
        '<p class="status-label">📖 Reading your document...</p>',
        unsafe_allow_html=True,
    )
    progress_bar.progress(0.05)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    time.sleep(0.4)

    # ── Stage 2: Extract text ─────────────────────────────────────────────────
    status_text.markdown(
        '<p class="status-label">📖 Reading your document...</p>',
        unsafe_allow_html=True,
    )
    text = extract_text(tmp_path)
    progress_bar.progress(0.25)
    time.sleep(0.3)

    # ── Stage 3: Chunk ────────────────────────────────────────────────────────
    status_text.markdown(
        '<p class="status-label">✂️ Splitting into chunks...</p>',
        unsafe_allow_html=True,
    )
    progress_bar.progress(0.40)
    chunks = chunk_text(text)
    time.sleep(0.3)

    # ── Stage 4: Embed + FAISS ────────────────────────────────────────────────
    status_text.markdown(
        '<p class="status-label">🧠 Building knowledge index...</p>',
        unsafe_allow_html=True,
    )
    progress_bar.progress(0.55)
    vs = build_vectorstore(chunks, api_key)
    save_vectorstore(vs, "vector_index/")
    progress_bar.progress(0.75)
    time.sleep(0.2)

    # ── Stage 5: Build chain ──────────────────────────────────────────────────
    status_text.markdown(
        '<p class="status-label">⚡ Connecting to Gemini...</p>',
        unsafe_allow_html=True,
    )
    progress_bar.progress(0.88)
    chain = build_qa_chain(api_key)
    time.sleep(0.2)

    # ── Done ──────────────────────────────────────────────────────────────────
    status_text.markdown(
        '<p class="status-label">✅ Ready to chat!</p>',
        unsafe_allow_html=True,
    )
    progress_bar.progress(1.0)
    time.sleep(0.6)

    # Persist to session state
    st.session_state.qa_chain   = chain
    st.session_state.filename   = uploaded_file.name
    st.session_state.ready      = True
    st.session_state.chat_history = []

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 1 — Upload Screen
# ─────────────────────────────────────────────────────────────────────────────
def render_upload_screen() -> None:
    """Render the centered upload landing screen with animated hero icon."""

    if not api_key:
        st.error(
            "⚠️ **GOOGLE_API_KEY not found.**  "
            "Add it to your `.env` file or Streamlit Cloud secrets."
        )
        st.stop()

    # Vertically space from top
    st.markdown("<div style='height: 8vh'></div>", unsafe_allow_html=True)

    # Centered column layout — no dead space
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Animated hero SVG icon
        st.markdown(HERO_SVG, unsafe_allow_html=True)

        # Headline with animated gradient on "PDF"
        st.markdown(
            '<h1 class="hero-title">Chat with your <span class="gradient-word">PDF</span></h1>',
            unsafe_allow_html=True,
        )

        # Subheadline
        st.markdown(
            '<p class="hero-sub">Upload any document. Ask anything. Get instant answers.</p>',
            unsafe_allow_html=True,
        )

        # Upload card — glassmorphism with gradient border
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop your PDF here",
            type=["pdf"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="upload-hint">📎 Drag & drop or click to browse</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Process uploaded file
        if uploaded is not None:
            st.markdown('<div class="status-card">', unsafe_allow_html=True)
            process_pdf(uploaded)
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 2 — Chat Screen
# ─────────────────────────────────────────────────────────────────────────────
def render_chat_screen() -> None:
    """Render the chat interface with top bar, memory badge, message history, and input."""

    # ── Top bar ───────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="top-bar">
          <div class="top-bar-left">
            <span class="doc-icon">📄</span>
            <span class="doc-name">{st.session_state.filename}</span>
            <span class="memory-badge">🧠 Memory on</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # "Upload new PDF" button — resets everything
    _, btn_col = st.columns([6, 1])
    with btn_col:
        if st.button("🔄 New PDF", key="reset_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            sources_text = (
                f'<p class="bubble-sources">📚 Sources: {msg["sources"]} chunk(s) used</p>'
                if msg.get("sources", 0) > 0
                else ""
            )
            st.markdown(
                f'<div class="assistant-bubble">{msg["content"]}{sources_text}</div>',
                unsafe_allow_html=True,
            )

    # Thinking dots while answer is generating
    if st.session_state.thinking:
        st.markdown(
            """
            <div class="thinking-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Inject auto-scroll script during generation
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                const interval = setInterval(() => {
                    const el = window.parent.document.querySelector('.main') || window.parent.document.documentElement;
                    if (el) {
                        el.scrollTop = el.scrollHeight;
                    }
                }, 100);
                window.onunload = () => clearInterval(interval);
            </script>
            """,
            height=0,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about your document...")

    if question and not st.session_state.thinking:
        # Append user message immediately
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        st.session_state.thinking = True
        st.rerun()


def stream_answer(question: str) -> None:
    """
    Call the RAG chain with memory, then stream the answer with a typewriter effect.

    Uses ``st.empty()`` to update a single placeholder character-by-character.
    Source count is appended below the answer once complete.

    Args:
        question: The user's natural-language question string.
    """
    result = ask_with_memory(
        st.session_state.qa_chain,
        question,
        st.session_state.chat_history,  # pass full history for memory
    )
    answer    = result["answer"]
    n_sources = len(result["sources"])

    # Typewriter — build up the displayed text one char at a time
    placeholder = st.empty()
    displayed   = ""
    for char in answer:
        displayed += char
        placeholder.markdown(
            f'<div class="assistant-bubble">{displayed}▍</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.012)

    # Final render with cursor removed + sources
    sources_html = (
        f'<p class="bubble-sources">📚 Sources: {n_sources} chunk(s) used</p>'
        if n_sources > 0
        else ""
    )
    placeholder.markdown(
        f'<div class="assistant-bubble">{answer}{sources_html}</div>',
        unsafe_allow_html=True,
    )

    # Persist to history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": n_sources}
    )
    st.session_state.thinking = False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Route to the correct screen and handle the thinking → answer transition."""
    if not st.session_state.ready:
        render_upload_screen()
    else:
        render_chat_screen()

        # If we are in thinking state, the last item in history is a user msg
        # → generate the answer now (this runs *after* the rerun triggered above)
        if (
            st.session_state.thinking
            and st.session_state.chat_history
            and st.session_state.chat_history[-1]["role"] == "user"
        ):
            question = st.session_state.chat_history[-1]["content"]
            stream_answer(question)
            st.rerun()


if __name__ == "__main__":
    main()
