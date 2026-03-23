"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 0.7rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    color: white;
}
.main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }
.main-header p  { margin: 0.1rem 0 0 0; opacity: 0.8; font-size: 0.8rem; }

.chunk-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.chunk-card:hover { border-color: #3b82f6; }

.meta-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Fira Code', monospace;
    margin-right: 6px;
}
.badge-topic      { background: #dbeafe; color: #1e40af; }
.badge-difficulty  { background: #dcfce7; color: #166534; }
.badge-type        { background: #fef3c7; color: #92400e; }

section[data-testid="stSidebar"] { background: #f1f5f9; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; border-radius: 8px;
}

.doc-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.doc-item strong { color: #1e293b; }
.doc-item .doc-meta { color: #64748b; font-size: 0.78rem; }

.stat-card {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: white;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-card .stat-value { font-size: 1.6rem; font-weight: 700; }
.stat-card .stat-label {
    font-size: 0.75rem; opacity: 0.8;
    text-transform: uppercase; letter-spacing: 0.5px;
}

.source-chip {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'Fira Code', monospace;
    margin: 2px 4px 2px 0;
}

div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.8rem;
}
.stSelectbox label, .stFileUploader label {
    font-weight: 600; color: #334155;
}

/* ---------- Layout: no main scroll, panels scroll internally ---------- */
.stMainBlockContainer {
    padding-top: 1rem !important;
    overflow: hidden !important;
}
section.main > div.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
}
/* Hide the main page scrollbar */
section.main {
    overflow: hidden !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    st.sidebar.markdown("### 📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "⬆️ Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        with st.sidebar.spinner("Chunking and ingesting..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = Path(tmp_dir) / uploaded_file.name
                    file_path.write_bytes(uploaded_file.getvalue())
                    file_paths.append(file_path)

                chunks = chunker.chunk_files(file_paths)
                result = store.ingest(chunks)

                st.session_state["last_ingestion_result"] = result
                if result.ingested > 0:
                    st.sidebar.success(
                        f"✅ {result.ingested} chunks added, "
                        f"{result.skipped} duplicates skipped"
                    )
                elif result.skipped > 0:
                    st.sidebar.warning(
                        f"⚠️ All {result.skipped} chunks already exist (duplicates)"
                    )
                if result.errors:
                    for err in result.errors:
                        st.sidebar.error(err)

    docs = store.list_documents()
    st.session_state["ingested_documents"] = docs

    if docs:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Ingested Documents")
        for doc in docs:
            col_name, col_del = st.sidebar.columns([5, 1])
            with col_name:
                st.sidebar.markdown(
                    f"<div class='doc-item'>"
                    f"<strong>{doc['source']}</strong><br>"
                    f"<span class='doc-meta'>"
                    f"<span class='meta-badge badge-topic'>{doc['topic']}</span> "
                    f"{doc['chunk_count']} chunks"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{doc['source']}"):
                    store.delete_document(doc["source"])
                    st.rerun()
    else:
        st.sidebar.info("Upload .pdf or .md files to populate the corpus.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    try:
        stats = store.get_collection_stats()
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Corpus Health")

        st.sidebar.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-value'>{stats['total_chunks']}</div>"
            f"<div class='stat-label'>Total Chunks</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if stats["topics"]:
            topic_badges = " ".join(
                f"<span class='meta-badge badge-topic'>{t}</span>"
                for t in stats["topics"]
            )
            st.sidebar.markdown(
                f"**Topics:** {topic_badges}", unsafe_allow_html=True
            )

        if stats["bonus_topics_present"]:
            st.sidebar.success("✅ Bonus topics present")
        else:
            st.sidebar.caption("⚠️ No bonus topics yet")
    except Exception:
        st.sidebar.warning("Could not load corpus stats.")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    st.markdown("#### 📄 Document Viewer")

    docs = st.session_state.get("ingested_documents", [])

    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    source_options = [doc["source"] for doc in docs]
    selected = st.selectbox("Select document", options=source_options)
    st.session_state["selected_document"] = selected

    if selected:
        chunks = store.get_document_chunks(selected)
        st.caption(f"📑 {len(chunks)} chunks from **{selected}**")

        chunk_container = st.container(height=380)
        with chunk_container:
            for i, chunk in enumerate(chunks):
                meta = chunk.metadata
                text_preview = chunk.chunk_text[:500]
                if len(chunk.chunk_text) > 500:
                    text_preview += "..."

                st.markdown(
                    f"<div class='chunk-card'>"
                    f"<div style='margin-bottom:6px;'>"
                    f"<strong>Chunk {i + 1}</strong> &nbsp;"
                    f"<span class='meta-badge badge-topic'>{meta.topic}</span>"
                    f"<span class='meta-badge badge-difficulty'>{meta.difficulty}</span>"
                    f"<span class='meta-badge badge-type'>{meta.type}</span>"
                    f"</div>"
                    f"<div style='font-size:0.9rem;color:#475569;line-height:1.5;'>"
                    f"{text_preview}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    st.markdown("#### 💬 Interview Prep Chat")

    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic_options = ["All"] + sorted(
            set(d["topic"] for d in st.session_state.get("ingested_documents", []))
        )
        topic_choice = st.selectbox("Topic Filter", options=topic_options)
        st.session_state["topic_filter"] = (
            None if topic_choice == "All" else topic_choice
        )
    with col_diff:
        diff_options = ["All", "beginner", "intermediate", "advanced"]
        diff_choice = st.selectbox("Difficulty Filter", options=diff_options)
        st.session_state["difficulty_filter"] = (
            None if diff_choice == "All" else diff_choice
        )

    chat_container = st.container(height=380)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='text-align:center;padding:3rem 1rem;color:#94a3b8;'>"
                "<div style='font-size:2.5rem;margin-bottom:0.5rem;'>🧠</div>"
                "<div style='font-size:1rem;font-weight:600;'>Ready to prep</div>"
                "<div style='font-size:0.85rem;'>"
                "Ask a deep learning question to get started</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        chips = " ".join(
                            f"<span class='source-chip'>{s}</span>"
                            for s in message["sources"]
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                if message.get("response_time"):
                    st.caption(f"⏱️ {message['response_time']:.1f}s")

                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        graph_input = {
            "messages": [HumanMessage(content=query)],
            "topic_filter": st.session_state["topic_filter"],
            "difficulty_filter": st.session_state["difficulty_filter"],
        }
        config = {"configurable": {"thread_id": f"session-{uuid.uuid4().hex[:8]}"}}

        with st.spinner("Thinking..."):
            start_time = time.time()
            try:
                result = graph.invoke(graph_input, config=config)
                elapsed = time.time() - start_time
                response = result.get("final_response")

                if response:
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "no_context_found": response.no_context_found,
                        "response_time": elapsed,
                    }
                else:
                    assistant_msg = {
                        "role": "assistant",
                        "content": "Sorry, I could not generate a response.",
                        "sources": [],
                        "no_context_found": True,
                        "response_time": elapsed,
                    }

            except Exception as e:
                elapsed = time.time() - start_time
                assistant_msg = {
                    "role": "assistant",
                    "content": f"An error occurred: {str(e)}",
                    "sources": [],
                    "no_context_found": False,
                    "response_time": elapsed,
                }

        st.session_state.chat_history.append(assistant_msg)
        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        "<div class='main-header'>"
        "<h1>🧠 Deep Learning Interview Prep Agent</h1>"
        "<p>RAG-powered interview preparation — "
        "built with LangChain, LangGraph, and ChromaDB</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()