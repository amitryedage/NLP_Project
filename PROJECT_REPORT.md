# 📘 NLP Visual Lab — Project Report

**Course Title:** Natural Language Processing (1ADPC402)  
**Academic Year:** 2025 – 2026  
**Department:** Artificial Intelligence and Data Science  
**Institution:** Annasaheb Dange College of Engineering and Technology (ADCET), Ashta  

---

## 1. Introduction
Natural Language Processing (NLP) is a core subfield of Artificial Intelligence concerned with facilitating intuitive, computational interactions between computers and human natural language. Foundational concepts in NLP—such as token segmentation, morphological stemming, lexical lemmatization, grammatical tagging, named entity identification, document vectorization, and dense vector embeddings—are essential building blocks for modern Large Language Models (LLMs) and search engines.

**NLP Visual Lab** is an interactive, browser-based visual intuition platform developed to provide students, researchers, and instructors with a live laboratory environment. Users can input arbitrary text and observe each core NLP transformation execute in real-time, backed by established open-source NLP libraries.

---

## 2. Problem Statement
NLP concepts are traditionally introduced through abstract mathematical equations and command-line code snippets. Novices and students frequently face obstacles in building strong conceptual intuition regarding:
1. How token boundaries are established across contractions and punctuation.
2. The critical operational difference between heuristic stemmers (which produce non-dictionary word roots) and lemmatizers.
3. How sequence models resolve grammatical part-of-speech ambiguity based on context.
4. How high-dimensional text is mapped into term-importance matrices (TF-IDF) and dense vector spaces (Word2Vec / GloVe).

Existing tools often rely on static diagrams, pre-computed examples, or fragmented scripts. There is a strong need for an integrated, single-dashboard laboratory environment running real NLP algorithms live on user-supplied text.

---

## 3. Project Objectives
* **Educational Clarity:** Deliver intuitive, graphical representations for 6 fundamental NLP concepts.
* **Real-Time Execution:** Execute real NLP models (spaCy, NLTK, scikit-learn) dynamically on user input without pre-rendered mockups.
* **Aesthetic & Accessible UI:** Maintain a clean, high-contrast **Light Theme** adhering to strict color tokens and WCAG contrast standards.
* **Reproducibility & Modern Tooling:** Manage dependencies and Python environments via Astral `uv` for zero-friction setup and fast virtual environment synchronization.
* **Modular Engineering:** Decouple user interface orchestration (`app.py`) from independent NLP computation modules (`modules/`).

---

## 4. Methodology & Architectural Design

### 4.1 System Architecture
The application is built on a decoupled, modular architectural pattern:

```
NLP_Project/
├── .streamlit/config.toml     # Light theme specification (base="light")
├── pyproject.toml            # Astral uv dependency management
├── app.py                    # Main dashboard controller & global state
├── modules/                  # Isolated NLP computational units
│   ├── tokenization.py       # Tab 1: Tokenization logic & Devanagari demo
│   ├── stemming_lemma.py     # Tab 2: Porter, Snowball & WordNet comparison
│   ├── pos_ner.py            # Tabs 3 & 4: spaCy POS tagging & DisplaCy NER
│   ├── tfidf.py              # Tab 5: BoW & TF-IDF matrices & Plotly heatmap
│   └── embeddings.py         # Tab 6: 300D word vectors & PCA 2D projection
├── utils/                    # Reusable services
│   ├── theme.py              # Light palette tokens, CSS & HTML chip generator
│   └── nlp_loader.py         # @st.cache_resource model & corpus loader
└── tests/                    # Pytest unit verification suite
```

### 4.2 Tech Stack Justification
* **Python 3.12 & Astral `uv`:** Provides lightning-fast resolution and deterministic lockfiles. Direct wheel installation for spaCy pre-trained models eliminates pip-wrapper overhead.
* **Streamlit:** Delivers a reactive web frontend with `@st.cache_resource` for zero-latency re-runs after initial model caching.
* **spaCy (`en_core_web_md`):** Industry-standard statistical NLP pipeline providing 300-dimensional Word2Vec vectors, transition-based POS parsing, and named entity extraction.
* **NLTK:** Proven heuristic implementations for Porter/Snowball stemmers and lexical WordNet lookup.
* **scikit-learn:** Efficient sparse matrix operations for `CountVectorizer`, `TfidfVectorizer`, and `PCA`.
* **Plotly:** Interactive, responsive vector and matrix visualizations styled with `plotly_white`.

---

## 5. Detailed Module Descriptions

### Module 1: Tokenization (`modules/tokenization.py`)
* **Purpose:** Demonstrates splitting continuous text into discrete grammatical units.
* **Algorithms:** NLTK Word Tokenizer (Penn Treebank conventions), NLTK Sentence Tokenizer (Punkt unsupervised model), and spaCy rule-based tokenizer.
* **Visuals:** Alternating pastel chips (`chip()`), token frequency counts, Lexical Diversity gauge (Type-Token Ratio - TTR), interactive N-Gram Phrase Inspector (Unigrams, Bigrams, Trigrams), indexed breakdown tables, and a bonus Devanagari (Marathi/Hindi) Unicode segmentation demo.

### Module 2: Stemming vs. Lemmatization (`modules/stemming_lemma.py`)
* **Purpose:** Contrasts heuristic suffix stripping with lexical vocabulary reduction.
* **Algorithms:** Porter Stemmer, Snowball Stemmer (English), and NLTK WordNet Lemmatizer (noun & verb morphological base lookup).
* **Visuals:** Side-by-side comparison tables color-coding discrepancies (`⚠️ Differ` in pastel pink, `✅ Match` in pastel green), with visual callout chips for irregular plurals (e.g., `mice` $\rightarrow$ `mouse`, `leaves` $\rightarrow$ `leaf`, `studies` $\rightarrow$ `study`).

### Module 3: Part-of-Speech Tagging (`modules/pos_ner.py`)
* **Purpose:** Assigns universal grammatical classes, fine-grained Penn Treebank tags, and syntactic dependencies.
* **Algorithms:** spaCy transition-based dependency parser and tagger.
* **Visuals:** Dynamic color legend matching active tags, color-coded word chips with POS sub-labels, Plotly frequency distribution bar chart, interactive **Syntactic Dependency Parse Tree** (DisplaCy `style="dep"`), and detailed definitions via `spacy.explain()`.

### Module 4: Named Entity Recognition (`modules/pos_ner.py`)
* **Purpose:** Identifies and classifies real-world entities into predefined semantic classes.
* **Algorithms:** spaCy pre-trained NER pipeline.
* **Visuals:** Custom light-themed `spacy.displacy` inline visualizer, extracted entity tables with character boundary offsets, and category breakdown chips.

### Module 5: TF-IDF & Bag of Words (`modules/tfidf.py`)
* **Purpose:** Visualizes multi-document vectorization and inverse term frequency weighting.
* **Algorithms:** scikit-learn `CountVectorizer` and `TfidfVectorizer` (with optional stopword removal).
* **Visuals:** High-density Plotly Heatmap (`Blues` colorscale) mapping vocabulary words against documents, interactive keyword selector, and per-document horizontal bar charts.

### Module 6: Word Embeddings & Vector Space (`modules/embeddings.py`)
* **Purpose:** Illustrates semantic distance and geometric vector arithmetic in continuous space.
* **Algorithms:** spaCy 300D static vectors from `en_core_web_md`, Principal Component Analysis (PCA $300\text{D} \rightarrow 2\text{D} / 3\text{D}$), and Cosine Similarity.
* **Visuals:** Interactive **2D Scatter vs. 3D Vector Space Orbit** toggle (`px.scatter_3d`), semantic trajectory shift arrows, Out-of-Vocabulary (OOV) error banners, live Cosine Similarity calculator with qualitative thresholds (High, Moderate, Low), a live **Semantic Vector Arithmetic Playground** ($\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$), and an interactive **2D Analogy Vector Geometry (Parallelogram Law) Visualizer** with directional parallelism metrics and displacement arrows.

---

## 6. Experimental Results & Verification
* **Unit Testing:** Comprehensive test suite in `tests/test_modules.py` verified via `uv run pytest`:
  * Validates chip HTML contrast and escaping.
  * Asserts contraction separation (`Don't` $\rightarrow$ `Do` + `n't`).
  * Asserts stemmer pseudo-roots (`studi`) vs. lemmatizer lexemes (`study`).
  * Asserts entity recognition on standard benchmarks (`Elon Musk` $\rightarrow$ `PERSON`, `Mumbai` $\rightarrow$ `GPE`).
  * Asserts TF-IDF matrix shape integrity and normalization ($\le 1.0$).
  * Asserts 300D vector dimensionality and cosine similarity hierarchy ($\text{sim}(\text{king}, \text{queen}) > \text{sim}(\text{king}, \text{apple})$).
  * Asserts analogy parallelogram vector geometry, PCA projection linearity, and high directional parallelism ($> 75\%$).

---

## 7. Limitations
1. **Pre-Trained Vocabulary Boundary:** Word embeddings rely on spaCy's static `en_core_web_md` vocabulary; highly specialized domain jargon or rare misspelled terms produce Out-of-Vocabulary warnings.
2. **Corpus Size for In-Browser Heatmaps:** Heatmap rendering is optimized for micro-corpora (3 to 15 documents); displaying thousands of documents simultaneously in a client browser is memory-intensive.
3. **No Dynamic Fine-Tuning:** By design (to maintain a lightweight, micro-project scope), the system does not train or fine-tune neural weights on user data.

---

## 8. Future Scope
* **Transformer Attention Visualizer:** Incorporate an attention-head heatmap using lightweight models (e.g., DistilBERT) to demonstrate self-attention weights.
* **Cross-Lingual Multilingual Embeddings:** Extend the 2D PCA vector space to multilingual models (e.g., comparing Marathi/Hindi words with English equivalents in shared vector space).
* **Interactive Dependency Parse Tree:** Add graphical dependency parse tree visualization using DisplaCy's `dep` mode.

---

## 9. References
1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media.
2. Honnibal, M., & Montani, I. (2017). *spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing*.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
4. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781.
5. Streamlit Documentation (2024). *Streamlit Python Framework & Configuration Guides*. https://docs.streamlit.io.
