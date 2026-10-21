# 🔬 NLP Visual Lab

An interactive educational web application designed for students and NLP beginners to *visualize* and interactively explore fundamental Natural Language Processing pipelines live on custom input text.

* **Course:** Natural Language Processing (1ADPC402)
* **Department:** Artificial Intelligence and Data Science
* **Institution:** Annasaheb Dange College of Engineering and Technology (ADCET), Ashta
* **Theme:** Strict Light Theme
* **Tech Stack:** Python 3.12, Astral `uv`, Streamlit, spaCy (`en_core_web_md`), NLTK, scikit-learn, Plotly, pandas, pytest

---

## 📑 Academic Documentation & Presentation Assets
* 📘 **[Project Report](PROJECT_REPORT.md):** Complete academic micro-project report (Introduction, Problem Statement, Objectives, Methodology, Results, Limitations, Future Scope, References).
* 🎤 **[Demo Script](DEMO_SCRIPT.md):** 3-to-5 minute live demonstration walkthrough with universal benchmark test sentences.

---

## 🚀 Quickstart Guide

This project is strictly managed via **Astral `uv`**. No direct `pip` installation is required.

### 1. Prerequisites
Install `uv` on your operating system if you haven't already:
* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

### 2. Environment Setup & Dependency Installation

Clone the repository and sync the virtual environment:

```bash
# Sync all dependencies including spaCy and its pre-trained model
uv sync
```

Download required NLTK corpora once:
```bash
uv run python -c "import nltk; [nltk.download(x) for x in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']]"
```

---

### 3. Running the Application

Launch the Streamlit interactive dashboard:

```bash
uv run streamlit run app.py
```

The application will start automatically in your default browser at `http://localhost:8501`.

---

### 4. Running Unit Tests

Run test suites using pytest via `uv`:

```bash
uv run pytest
```

---

## 🗂️ Project Structure

```
NLP_Project/
├── .streamlit/
│   └── config.toml          # Enforces light theme specification
├── .python-version          # Pinned to Python 3.12
├── pyproject.toml           # uv project configuration and dependencies
├── app.py                   # Main entry point with 6 interactive tabs & global presets
├── modules/
│   ├── __init__.py
│   ├── tokenization.py      # Tab 1: Word, Sentence, spaCy tokenization + Devanagari
│   ├── stemming_lemma.py    # Tab 2: Porter, Snowball, WordNet lemmatizer
│   ├── pos_ner.py           # Tabs 3 & 4: POS tagging & DisplaCy NER
│   ├── tfidf.py             # Tab 5: BoW and TF-IDF matrices & heatmaps
│   └── embeddings.py        # Tab 6: 2D PCA word vectors & cosine similarity
├── utils/
│   ├── __init__.py
│   ├── theme.py             # Pastel chip generator, colors & custom CSS
│   └── nlp_loader.py        # Streamlit-cached spaCy and NLTK resource loader
├── tests/
│   └── test_modules.py      # Pytest module sanity tests
├── PROJECT_REPORT.md        # Academic submission report
├── DEMO_SCRIPT.md           # 3-5 minute presentation script
├── PRD_NLP_Visual_Lab.md    # Product Requirements Document
└── README.md
```

---

## 📋 Features Overview

1. **✂️ Tokenization:** Live comparison between NLTK Word, NLTK Sentence, and spaCy Tokenizers with pastel chips and indexed tables. Includes bonus Unicode-aware Devanagari (Marathi/Hindi) tokenization.
2. **🌱 Stemming vs. Lemmatization:** Side-by-side comparison of Porter, Snowball, and WordNet with highlighting of inflectional differences.
3. **🏷️ POS Tagging:** Color-coded grammatical tags with human-readable explanations via `spacy.explain()`.
4. **🔍 Named Entity Recognition (NER):** DisplaCy visualizer for PERSON, ORG, GPE, DATE, and more.
5. **📊 TF-IDF & Bag of Words:** Numerical matrix inspector with one-click CSV export, Document-Word heatmap, **Pairwise Inter-Document Similarity Heatmap**, and a live **Information Retrieval (Search Engine Simulator)** that ranks corpus documents against user queries using cosine similarity.
6. **🌌 Word Embeddings:** Interactive 2D PCA projection with semantic trajectory shift arrows, 3D vector space orbit, live cosine similarity comparator, **Top-$K$ Nearest Semantic Neighbors Explorer** with similarity ranking charts, **Parallelogram Law Vector Geometry visualizer** ($\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$), and 2D coordinates CSV export.
7. **⚡ Global Example Presets:** One-click benchmark sentence loader across tabs.

---

## ☁️ Optional Deployment (Streamlit Community Cloud)

If deploying to Streamlit Community Cloud:
```bash
# Export requirements.txt if needed by cloud platform
uv pip compile pyproject.toml -o requirements.txt
```
Connect your GitHub repository to [share.streamlit.io](https://share.streamlit.io) and select `app.py` as the entry file.
