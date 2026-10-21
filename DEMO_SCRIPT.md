# 🎤 NLP Visual Lab — 3 to 5 Minute Demo Script

**Course:** Natural Language Processing (1ADPC402)  
**Department:** Artificial Intelligence & Data Science, ADCET Ashta  
**Project:** NLP Visual Lab  
**Presentation Target:** 3 to 5 minutes  

---

## 🎯 Demo Preparation Checklist
1. Launch the app in your terminal:
   ```bash
   uv run streamlit run app.py
   ```
2. Verify browser opens at `http://localhost:8501` in the **Light Theme**.
3. Keep the "⚡ Try Quick Preset Sentences" expander at the top in view.

---

## ⏱️ Minute-by-Minute Presentation Plan

### 0:00 – 0:45 | Introduction & Problem Statement
* **Speaker:**
  > *"Good morning/afternoon, evaluators and classmates. Today we are demonstrating **NLP Visual Lab**, an interactive visual intuition engine for Natural Language Processing.*
  >
  > *Most NLP concepts are taught through dry textbook definitions or static console printouts. Beginners struggle to visualize what happens to words inside an NLP pipeline.*
  >
  > *We built a unified, light-themed Streamlit application powered by real NLP libraries—spaCy, NLTK, and scikit-learn. Nothing here is pre-baked or static; every transformation runs live on user input."*

---

### 0:45 – 1:30 | Tab 1: Tokenization & Contraction Splitting
* **Action:** Click on the **✂️ Tokenization** tab. Highlight the live pastel chips.
* **Speaker:**
  > *"First is **Tokenization**, the foundational step where continuous text is segmented into discrete computational units.*
  >
  > *Notice our live token stream rendered as color-coded chips with guaranteed high-contrast dark text. If we switch between **NLTK Word Tokenizer** and **Sentence Tokenizer**, we can see how contractions like `Don't` are split into `Do` and `n't`, separating the auxiliary verb from the negative particle.*
  >
  > *(Bonus) We also included a Unicode-aware Devanagari tokenization demonstration for Marathi and Hindi scripts."*

---

### 1:30 – 2:15 | Tab 2: Stemming vs. Lemmatization
* **Action:** Click on **🌱 Stem / Lemma** tab.
* **Speaker:**
  > *"Next, we compare morphological reduction: **Stemming vs. Lemmatization**.*
  >
  > *Stemmers like Porter and Snowball use heuristic suffix stripping. Notice in our side-by-side table that `studies` is chopped into the pseudo-word `studi`, and `flies` becomes `fli`.*
  >
  > *In contrast, the WordNet Lemmatizer uses vocabulary databases and grammar rules to return true dictionary base words: `study` and `fly`.*
  >
  > *Our app automatically highlights discrepancies in pastel pink, showing students exactly where and why stemmers fail on irregular inflections like `mice` and `feet`."*

---

### 2:15 – 3:00 | Tabs 3 & 4: POS Tagging, Syntactic Dependency Tree & DisplaCy NER
* **Action:** Click on **🏷️ POS Tags**, scroll to the **Syntactic Dependency Tree**, then switch to **🔍 NER**.
* **Speaker:**
  > *"In **Tab 3**, we use spaCy's statistical tagger to classify every word into its grammatical role (Noun, Verb, Adjective, etc.) with a live Plotly frequency bar chart and human-readable Penn Treebank explanations.*
  >
  > *Scroll down to our live **Syntactic Dependency Parse Tree**, which renders curved grammatical arcs connecting Subject, Root Verb, and Direct Object.*
  >
  > *Moving to **Tab 4: Named Entity Recognition**, we integrate spaCy's DisplaCy visualizer styled with our light pastel palette. On our benchmark sentence:*
  > `'Elon Musk visited Mumbai on 5 May 2024 to announce Tesla's $2 billion investment...'`
  > *The model extracts `Elon Musk` as a PERSON, `Mumbai` as a GPE (geopolitical entity), `Tesla` as an ORG, and `$2 billion` as MONEY, complete with start-end character spans."*

---

### 3:00 – 3:55 | Tabs 5 & 6: TF-IDF, 3D Orbit & Vector Arithmetic
* **Action:** Click on **📊 TF-IDF**, show the heatmap, then switch to **🌌 Embeddings**. Point out the **Semantic Trajectory Arrows** in 2D PCA, toggle to **3D Interactive Orbit** and spin with mouse, then scroll down to the **Vector Arithmetic Solver** and the **Parallelogram Law Visualizer**.
* **Speaker:**
  > *"In **Tab 5**, we show how text becomes numerical matrices. Our **TF-IDF Heatmap** visualizes word importance across documents. Notice how common words receive lower weights, while rare, discriminative words shine as top keywords.*
  >
  > *Finally, in **Tab 6: Word Embeddings**, we extract 300-dimensional dense vectors from spaCy's `en_core_web_md`.*
  >
  > *In the 2D projection, our **Semantic Trajectory Arrows** visually link analogical pairs like `Man ➔ King` and `Woman ➔ Queen`. You can also toggle to our **3D Interactive Orbit** to freely rotate and zoom into semantic clusters.*
  >
  > *Even more exciting is our **Semantic Vector Arithmetic Playground**: when we compute $\text{King} - \text{Man} + \text{Woman}$, the model mathematically solves for **Queen** at the top of the leaderboard with a 78% similarity match!*
  >
  > *Scroll down to our live **Parallelogram Law Visualizer**: it plots the 4 words and draws the two displacement vectors $\vec{\Delta}_1$ and $\vec{\Delta}_2$. The app measures an astonishing **94%+ directional parallelism**, geometrically proving that Word2Vec preserves semantic shifts in vector space!"*

---

### 3:50 – 4:15 | Conclusion & Technical Rigor
* **Speaker:**
  > *"In summary, NLP Visual Lab bridges the gap between theory and intuition. The entire codebase is strictly managed via Astral `uv`, follows a modular architecture where `app.py` only orchestrates standalone modules, and enforces a clean, accessible light theme.*
  >
  > *Thank you, and we welcome any questions!"*

---

## 📌 3 Universal Benchmark Sentences for Live Testing

1. **Sentence 1 (General Linguistic & Contraction Pipeline):**
   > *"Natural Language Processing (NLP) enables computers to understand, interpret, and manipulate human speech and text. Don't you think it's fascinating?"*
2. **Sentence 2 (NER & Geopolitical Entity Extraction):**
   > *"Elon Musk visited Mumbai on 5 May 2024 to announce Tesla's $2 billion investment alongside Indian officials."*
3. **Sentence 3 (Morphology & Irregular Plurals):**
   > *"The experienced researchers were studying flying bats, green leaves, and various connections while running happily on their feet."*
