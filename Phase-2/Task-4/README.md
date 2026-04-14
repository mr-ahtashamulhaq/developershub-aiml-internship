# Context-Aware Chatbot Using LangChain & RAG

## Objective

Build a conversational chatbot capable of **remembering conversation context** and **retrieving accurate answers from a vectorized knowledge base** using Retrieval-Augmented Generation (RAG).

---

## Methodology / Approach

### Architecture

```
User Query
    │
    ▼
Streamlit UI (app.py)
    │
    ▼
ConversationalRetrievalChain (LangChain)
    ├── ConversationBufferMemory  ← stores full chat history
    ├── FAISS Vector Store        ← similarity search over Wikipedia corpus
    │       └── HuggingFace Embeddings (all-MiniLM-L6-v2)
    └── Groq LLM (llama3-8b-8192)
```

### Steps

1. **Corpus Ingestion** (`data_loader.py`)
   - Fetches 20 Wikipedia articles on AI/ML topics
   - Splits text into 1000-token chunks with 150-token overlap
   - Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`
   - Stores the FAISS index on disk (`./faiss_index/`)

2. **RAG Chain** (`chatbot.py`)
   - Loads FAISS index at startup
   - Uses `ChatGroq` (`llama3-8b-8192`) as the LLM
   - `ConversationsBufferMemory` preserves the full dialogue
   - Custom prompt condenses follow-up questions into standalone queries
   - Returns answer + source document URLs

3. **Streamlit UI** (`app.py`)
   - Gradient chat bubbles for user and bot messages
   - Wikipedia source citations displayed as chips
   - "Clear Conversation" button resets memory
   - Sidebar shows knowledge base topics and model details

---

## Knowledge Base Topics

| Topic | Wikipedia Page |
|-------|----------------|
| Artificial Intelligence | [link](https://en.wikipedia.org/wiki/Artificial_intelligence) |
| Machine Learning | [link](https://en.wikipedia.org/wiki/Machine_learning) |
| Deep Learning | [link](https://en.wikipedia.org/wiki/Deep_learning) |
| Natural Language Processing | [link](https://en.wikipedia.org/wiki/Natural_language_processing) |
| Neural Network | [link](https://en.wikipedia.org/wiki/Neural_network) |
| Reinforcement Learning | [link](https://en.wikipedia.org/wiki/Reinforcement_learning) |
| Computer Vision | [link](https://en.wikipedia.org/wiki/Computer_vision) |
| Large Language Model | [link](https://en.wikipedia.org/wiki/Large_language_model) |
| Transformer (ML) | [link](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) |
| GAN | [link](https://en.wikipedia.org/wiki/Generative_adversarial_network) |
| CNN | [link](https://en.wikipedia.org/wiki/Convolutional_neural_network) |
| RNN | [link](https://en.wikipedia.org/wiki/Recurrent_neural_network) |
| Transfer Learning | [link](https://en.wikipedia.org/wiki/Transfer_learning) |
| Unsupervised Learning | [link](https://en.wikipedia.org/wiki/Unsupervised_learning) |
| Supervised Learning | [link](https://en.wikipedia.org/wiki/Supervised_learning) |
| RAG | [link](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) |
| LangChain | [link](https://en.wikipedia.org/wiki/LangChain) |
| Chatbot | [link](https://en.wikipedia.org/wiki/Chatbot) |
| Word Embedding | [link](https://en.wikipedia.org/wiki/Word_embedding) |
| BERT | [link](https://en.wikipedia.org/wiki/BERT_(language_model)) |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A [Groq](https://console.groq.com/) API key (free tier is sufficient)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file (already provided) or set the environment variable:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Build the knowledge base (run once)

```bash
python data_loader.py
```

This downloads Wikipedia articles, creates embeddings, and saves the FAISS index to `./faiss_index/`.

### 4. Launch the chatbot

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Key Results / Observations

- **Context Memory**: The bot correctly references earlier parts of the conversation in follow-up answers (e.g., asking "What about its applications?" after "What is deep learning?" retrieves application-specific chunks).
- **Accurate Retrieval**: FAISS similarity search consistently retrieves the top-4 most relevant Wikipedia chunks, keeping answers grounded.
- **Source Transparency**: Every response includes clickable Wikipedia source links for verification.
- **Low Latency**: Groq's hardware inference delivers responses in under 2 seconds.
- **Honest Uncertainty**: Custom prompt instructs the LLM to acknowledge when an answer is outside the knowledge base.

---

## Skills Gained

- Conversational AI development with LangChain
- Document embedding and vector search (FAISS + HuggingFace)
- Retrieval-Augmented Generation (RAG)
- LLM integration via Groq API
- Streamlit deployment

---

## File Structure

```
Task-4/
├── app.py              # Streamlit chat UI
├── chatbot.py          # Core RAG chain with memory
├── data_loader.py      # Wikipedia corpus ingestion & FAISS builder
├── requirements.txt    # Python dependencies
├── .env                # API keys (do NOT commit to git)
├── faiss_index/        # Generated vector store (created by data_loader.py)
└── README.md           # This file
```
