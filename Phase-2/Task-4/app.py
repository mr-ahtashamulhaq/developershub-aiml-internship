"""
app.py
======
Streamlit UI for the Context-Aware RAG Chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st
from chatbot import load_vector_store, build_chain, get_response

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Context-Aware AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Global font & background ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Chat bubbles ── */
        .user-bubble {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            padding: 12px 18px;
            border-radius: 20px 20px 4px 20px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 4px 12px rgba(99,102,241,0.3);
        }
        .bot-bubble {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            color: #e2e8f0;
            padding: 12px 18px;
            border-radius: 20px 20px 20px 4px;
            margin: 8px 0;
            max-width: 85%;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            border-left: 3px solid #6366f1;
        }
        .source-chip {
            display: inline-block;
            background: #1e293b;
            color: #94a3b8;
            font-size: 0.72rem;
            padding: 2px 10px;
            border-radius: 99px;
            margin: 2px 4px 2px 0;
            border: 1px solid #334155;
        }
        .source-section {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #334155;
        }

        /* ── Header ── */
        .hero {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
        }
        .hero h1 { font-weight: 700; font-size: 2rem; letter-spacing: -0.5px; }
        .hero p  { color: #94a3b8; font-size: 0.95rem; }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        .stButton button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 **AI/ML Chatbot**")
    st.markdown("---")
    st.markdown(
        """
        **Knowledge Base Topics:**
        - Artificial Intelligence
        - Machine Learning & Deep Learning
        - NLP & Transformers
        - Neural Networks (CNN, RNN)
        - GANs & Embeddings
        - Reinforcement Learning
        - Computer Vision
        - Large Language Models
        - LangChain & RAG
        """
    )
    st.markdown("---")

    if st.button("🗑 Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chain = build_chain(st.session_state.vector_store)
        st.rerun()

    st.markdown("---")
    st.markdown("**Model:** `llama3-8b-8192` via Groq")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("**Vector Store:** FAISS")
    st.caption("Context-Aware RAG Chatbot — Task 4")


# ── Load vector store and chain (cached per session) ──────────────────────────
@st.cache_resource(show_spinner="⚙  Loading knowledge base…")
def init_resources():
    """Load vector store once and cache it for the lifetime of the app."""
    vs = load_vector_store()
    return vs


if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = init_resources()
        st.session_state.chain = build_chain(st.session_state.vector_store)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🤖 Context-Aware AI Chatbot</h1>
        <p>Powered by <b>LangChain</b> · <b>RAG</b> · <b>Groq LLaMA 3</b> · <b>FAISS</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">💬 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            chips = "".join(
                f'<a href="{s}" target="_blank" class="source-chip">📎 {s.split("/wiki/")[-1].replace("_", " ")}</a>'
                for s in msg["sources"]
            )
            sources_html = f'<div class="source-section">📚 <b>Sources:</b> {chips}</div>'
        st.markdown(
            f'<div class="bot-bubble">🤖 {msg["content"]}{sources_html}</div>',
            unsafe_allow_html=True,
        )

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask me anything about AI/ML…"):
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-bubble">💬 {prompt}</div>', unsafe_allow_html=True)

    # Get response
    with st.spinner("🔍 Retrieving and reasoning…"):
        result = get_response(st.session_state.chain, prompt)

    # Store and display assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )

    sources_html = ""
    if result["sources"]:
        chips = "".join(
            f'<a href="{s}" target="_blank" class="source-chip">📎 {s.split("/wiki/")[-1].replace("_", " ")}</a>'
            for s in result["sources"]
        )
        sources_html = f'<div class="source-section">📚 <b>Sources:</b> {chips}</div>'

    st.markdown(
        f'<div class="bot-bubble">🤖 {result["answer"]}{sources_html}</div>',
        unsafe_allow_html=True,
    )
