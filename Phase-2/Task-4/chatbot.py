"""
chatbot.py
==========
Core RAG chatbot module.
- Loads the FAISS vector store from disk
- Initialises ChatGroq LLM (llama3-8b-8192)
- Builds a ConversationalRetrievalChain with ConversationBufferMemory
- Exposes a simple get_response() function used by the Streamlit UI
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Custom prompts ─────────────────────────────────────────────────────────────
_CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the \
follow up question to be a standalone question that captures the full context.

Chat History:
{chat_history}

Follow Up Question: {question}
Standalone Question:"""
)

_QA_PROMPT = PromptTemplate.from_template(
    """You are a knowledgeable AI/ML assistant with access to a curated knowledge \
base. Use the retrieved context below to answer the question thoughtfully and \
concisely. If the answer is not in the context, say so honestly rather than \
making something up.

Context:
{context}

Question: {question}

Answer:"""
)


def load_vector_store() -> FAISS:
    """Load the persisted FAISS vector store from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}/'. "
            "Please run `python data_loader.py` first to build the knowledge base."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def build_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """Construct the ConversationalRetrievalChain."""
    # LLM — Groq (fast, free-tier friendly)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.3,
    )

    # Retriever — top-4 most similar chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Memory — keeps the full conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Build chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=_CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain


def get_response(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Send a question to the chain and return a dict with:
      - answer (str)
      - sources (list[str])
    """
    result = chain.invoke({"question": question})
    answer = result.get("answer", "I could not generate a response.")
    sources = list(
        {doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])}
    )
    return {"answer": answer, "sources": sources}
