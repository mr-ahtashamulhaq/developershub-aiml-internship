<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=6C63FF&center=true&vCenter=true&width=700&lines=AI%2FML+Engineering+Internship;DevelopersHub+Corporation;6+Tasks+%E2%80%A2+2+Phases+%E2%80%A2+Real+ML+Projects" alt="Typing SVG" />

<br/>

![Phase 1](https://img.shields.io/badge/Phase%201-3%20Tasks%20Completed-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)
![Phase 2](https://img.shields.io/badge/Phase%202-3%20Tasks%20Completed-6C63FF?style=for-the-badge&logo=checkmarx&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)

<br/>

**Intern:** Muhammad Ahtasham Ul Haq &nbsp;|&nbsp; **GitHub:** [@mr-ahtashamulhaq](https://github.com/mr-ahtashamulhaq) &nbsp;|&nbsp; **Organization:** DevelopersHub Corporation

</div>

---

## What's in this repo

Six end-to-end ML projects completed across two phases of the DevelopersHub AI/ML Engineering Internship — from basic EDA and classical ML to transformer fine-tuning, RAG pipelines, and LLM prompt engineering.

---

## 🗺️ Full Tasks Map

<div align="center">

| Phase | # | Task | Key Technique | Status |
|:---:|:---:|---|---|:---:|
| 1 | 1 | Iris Dataset EDA & Visualization | Exploratory Data Analysis | ✅ |
| 1 | 2 | Stock Price Prediction (AAPL) | Linear Regression · Random Forest | ✅ |
| 1 | 3 | House Price Prediction | Gradient Boosting · Feature Importance | ✅ |
| 2 | 4 | Context-Aware RAG Chatbot | LangChain · FAISS · Claude Haiku | ✅ |
| 2 | 5 | Auto Tagging Support Tickets | Prompt Engineering · Zero/Few-Shot | ✅ |
| 2 | 6 | Customer Churn ML Pipeline | Scikit-learn Pipeline · GridSearchCV | ✅ |

</div>

---

## 🟢 Phase 1 — Classical ML & EDA

### Task 1 — Iris Dataset EDA
`task1-iris-eda/`

Full exploratory analysis of the classic Iris dataset. Built a pairplot scatter matrix, feature histograms with KDE curves, box plots for outlier detection, and a correlation heatmap.

**Finding:** Petal length and petal width are the strongest species separators. Setosa is completely linearly separable from the other two classes.

**Stack:** `pandas` · `seaborn` · `matplotlib`

---

### Task 2 — Stock Price Prediction
`task2-stock-prediction/`

Fetched 5 years of Apple (AAPL) daily data via `yfinance`. Engineered features — 5-day/10-day moving averages, intraday price change, daily range — and trained two models to predict the next day's closing price.

**Finding:** Random Forest outperformed Linear Regression. Moving averages and the current close carried the most predictive signal.

**Stack:** `yfinance` · `scikit-learn` · `matplotlib`

---

### Task 3 — House Price Prediction
`task6-house-price/`

California Housing Dataset. Full EDA including a geographic price map, then trained Linear Regression and Gradient Boosting. Analyzed feature importance and residuals.

**Finding:** Gradient Boosting significantly outperformed Linear Regression (lower RMSE). Median income and geographic location are the two strongest price drivers.

**Stack:** `scikit-learn` · `seaborn` · `matplotlib`

---

## 🟣 Phase 2 — Advanced ML, LLMs & Pipelines

### Task 4 — Context-Aware RAG Chatbot
`Phase-2/Task-4/`

Built a fully conversational RAG chatbot over an AI/ML knowledge base (8 topic documents). Users can ask follow-up questions that reference prior turns — the model correctly resolves context without re-stating it.

**Architecture:**
```
User Query → Embed → FAISS Semantic Search → Top-K Chunks
                ↓                                    ↓
         Chat Memory (k=5)  →  Claude Haiku LLM  →  Answer
```

**Highlights:**
- `sentence-transformers/all-MiniLM-L6-v2` for local CPU embeddings
- `FAISS` vector index with cosine similarity
- `ConversationBufferWindowMemory` (5-turn window) for context tracking
- Semantic similarity heatmap shows retrieval correctly maps queries to relevant docs
- Deployed via **Streamlit** (`app.py`)

**Stack:** `LangChain` · `FAISS` · `sentence-transformers` · `Claude Haiku` · `Streamlit`

---

### Task 5 — Auto Tagging Support Tickets
`Phase-2/Task-5/`

LLM-powered auto-tagger that assigns the top 3 most relevant tags per ticket from a 36-tag taxonomy — no model training required. Compared **zero-shot** vs **few-shot** prompting on 25 real-world-style support tickets.

**Results:**

<div align="center">

| Method | Precision | Recall | F1 Score | Top-1 Hit Rate |
|:---:|:---:|:---:|:---:|:---:|
| Zero-Shot | 0.92 | 0.92 | 0.92 | 1.00 |
| **Few-Shot** | **0.99** | **0.99** | **0.99** | **1.00** |

</div>

**Finding:** Few-shot outperforms zero-shot by ~7 F1 points. Five labeled examples anchor the model's specificity and tag ordering. Top-1 accuracy is perfect for both — the primary category is always correct.

**Stack:** `Anthropic API` · `Claude Haiku` · `pandas` · `matplotlib`

---

### Task 6 — Customer Churn ML Pipeline
`Phase-2/Task-2/`

Production-ready, fully encapsulated ML pipeline for Telco customer churn prediction. A single `.joblib` file takes raw CSV input and returns predictions — no external preprocessing needed at inference time.

**Pipeline design:**
```
Raw Data
   ↓
ColumnTransformer
   ├── StandardScaler  →  numeric features
   └── OneHotEncoder   →  categorical features
   ↓
Model (Logistic Regression / Random Forest)
   ↓
Prediction
```

**Results:**

<div align="center">

| Model | Accuracy | F1 Score | ROC-AUC |
|:---:|:---:|:---:|:---:|
| Logistic Regression (Tuned) | 0.61 | 0.38 | 0.69 |
| Random Forest (Tuned) | 0.78 | 0.27 | 0.68 |

</div>

**Finding:** Class imbalance (~17% churn rate) is the core challenge. `class_weight='balanced'` recovers recall on the minority class. LR edges out RF on AUC. Top churn drivers: `MonthlyCharges`, `TotalCharges`, contract type, and `tenure`.

**Stack:** `scikit-learn` · `joblib` · `pandas` · `seaborn`

---

## 🛠️ Tech Stack — Full Picture

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic%20API-D97757?style=flat-square&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

## 📁 Repository Structure

```
mr-ahtashamulhaq-developershub-aiml-internship/
│
├── 📂 task1-iris-eda/
│   ├── task1_iris_eda.ipynb
│   └── README.md
│
├── 📂 task2-stock-prediction/
│   ├── task2_stock_prediction.ipynb
│   └── README.md
│
├── 📂 task6-house-price/
│   ├── task6_house_price_prediction.ipynb
│   └── README.md
│
├── 📂 Phase-2/
│   │
│   ├── 📂 Task-2/                        ← Churn ML Pipeline
│   │   ├── task2_churn_pipeline.ipynb
│   │   ├── telco_churn.csv
│   │   ├── churn_pipeline.joblib
│   │   └── README.md
│   │
│   ├── 📂 Task-4/                        ← RAG Chatbot
│   │   ├── Task4_RAG_Chatbot.ipynb
│   │   ├── app.py                        ← Streamlit UI
│   │   ├── chatbot.py
│   │   ├── data_loader.py
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── 📂 Task-5/                        ← Auto Ticket Tagging
│   │   ├── task5_auto_tagging.ipynb
│   │   ├── support_tickets.csv
│   │   ├── tagging_results.csv
│   │   └── README.md
│   │
│   └── README.md
│
└── README.md  ← You are here
```

---

## ▶️ How to Run

**Phase 1 notebooks** — open any `.ipynb` in [Google Colab](https://colab.research.google.com), click **Runtime → Run All**. Task 2 needs internet access for `yfinance` (Colab handles this automatically).

**Phase 2 notebooks** — run locally:
```bash
pip install -r Phase-2/Task-4/requirements.txt
jupyter notebook
```

**RAG Chatbot (Streamlit app):**
```bash
cd Phase-2/Task-4
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

---

<div align="center">

Made with 🔥 by **Muhammad Ahtasham Ul Haq**

[![GitHub](https://img.shields.io/badge/GitHub-mr--ahtashamulhaq-181717?style=for-the-badge&logo=github)](https://github.com/mr-ahtashamulhaq)

</div>