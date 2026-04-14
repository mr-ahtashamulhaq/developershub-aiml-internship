"""
data_loader.py
==============
Fetches Wikipedia articles on AI/ML topics, splits them into chunks,
creates HuggingFace embeddings, and saves a FAISS vector store to disk.

Run this script ONCE before launching the chatbot:
    python data_loader.py
"""

import os
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Topics to build the knowledge corpus ──────────────────────────────────────
TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Neural network",
    "Reinforcement learning",
    "Computer vision",
    "Large language model",
    "Transformer (machine learning model)",
    "Generative adversarial network",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transfer learning",
    "Unsupervised learning",
    "Supervised learning",
    "Retrieval-augmented generation",
    "LangChain",
    "Chatbot",
    "Word embedding",
    "BERT (language model)",
]

FAISS_INDEX_PATH = "faiss_index"


def fetch_wikipedia_articles(topics: list) -> list:
    """Download Wikipedia article summaries for each topic."""
    docs = []
    for topic in topics:
        try:
            print(f"  Fetching: {topic}")
            page = wikipedia.page(topic, auto_suggest=False)
            docs.append({"content": page.content, "source": page.url, "title": topic})
        except wikipedia.exceptions.DisambiguationError as e:
            # Take the first suggestion on disambiguation
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                docs.append({"content": page.content, "source": page.url, "title": topic})
            except Exception:
                print(f"    ⚠ Skipped (disambiguation): {topic}")
        except Exception as exc:
            print(f"    ⚠ Skipped ({exc}): {topic}")
    return docs


def build_vector_store(docs: list) -> FAISS:
    """Split documents and create a FAISS vector store."""

    # Convert raw dicts to LangChain Document objects
    lc_docs = [
        Document(page_content=d["content"], metadata={"source": d["source"], "title": d["title"]})
        for d in docs
    ]

    # Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(lc_docs)
    print(f"\n📄 Total chunks created: {len(chunks)}")

    # Create embeddings using a lightweight, free HuggingFace model
    print("⚙  Creating embeddings (this may take a minute)…")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Build and return FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def main():
    print("=" * 60)
    print("  Context-Aware Chatbot — Vector Store Builder")
    print("=" * 60)

    print("\n📥 Downloading Wikipedia articles…")
    docs = fetch_wikipedia_articles(TOPICS)
    print(f"✅ Fetched {len(docs)} articles.\n")

    vector_store = build_vector_store(docs)

    # Persist to disk
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"\n✅ FAISS index saved to './{FAISS_INDEX_PATH}/'")
    print("\nYou can now run the chatbot with:  streamlit run app.py")


if __name__ == "__main__":
    main()
