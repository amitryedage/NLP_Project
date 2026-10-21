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
  > *We built a unified, light-themed Streamlit application powered by real NLP libraries—spaCy, NLTK, and scikit-learn. Nothing here is pre-baked or static; every transformation runs live on user input.*
  >
  > *Right at the top, notice our **Corpus Profiler & Linguistic Readability Strip**: it computes total word and sentence counts, unique vocabulary size $|V|$, Type-Token Ratio (TTR) for lexical diversity, Hapax Legomena for vocabulary novelty, and the Flesch Reading Ease score live on our input text."*

---

### 0:45 – 1:30 | Tab 1: Tokenization, Stopwords, Zipf's Law & N-Gram Markov LM
* **Action:** Click on the **✂️ Tokenization** tab. Toggle **"Highlight English Stopwords"**, expand the **Hierarchical Word Treemap**, scroll down to the **Zipf's Law Distribution Plot**, showcase the **N-Gram Language Model** (pick a prompt word and click "Generate Continuation"), expand the **Devanagari Akshara Segmentation** demo, and point out the **Download CSV** button.
* **Speaker:**
  > *"First is **Tokenization**, the foundational step where continuous text is segmented into discrete computational units.*
  >
  > *Notice our live token stream rendered as color-coded chips with guaranteed high-contrast dark text. If we switch between **NLTK Word Tokenizer** and **Sentence Tokenizer**, we can see how contractions like `Don't` are split into `Do` and `n't`, separating the auxiliary verb from the negative particle.*
  >
  > *With our **Stopword Filter**, we can instantly isolate non-content grammatical filler words (`the`, `is`, `in`) with real-time volume reduction percentages.*
  >
  > *Expand our **Hierarchical Linguistic Word Treemap**: it replaces messy static word clouds with interactive rectangular clusters grouped by grammatical category and token length.*
  >
  > *Scroll down to our **Zipf's Law Linguistic Distribution Plot**: it plots empirical word frequency against rank on log-log coordinates, visually demonstrating that word frequency is inversely proportional to rank.*
  >
  > *Directly below, look at our **N-Gram Markov Language Model & Next-Word Predictor**: it calculates empirical transition probabilities $P(w_t \mid w_{t-1})$ with an optional Laplace (Add-1) smoothing toggle. When we select a prompt word like `'natural'`, it ranks candidate continuations like `'language'`, and clicking **'Generate Continuation'** lets us watch the Markov chain autonomously construct phrases token by token!*
  >
  > *(Bonus) We also included a Unicode-aware Devanagari Akshara and conjunct segmentation demo for Marathi and Hindi scripts, and one-click CSV export."*

---

### 1:30 – 2:15 | Tab 2: Stemming, Lemmatization, Edit Distance & Morpheme Anatomy
* **Action:** Click on **🌱 Stem / Lemma** tab. Showcase the discrepancy chips, styled comparison table, the **Minimum Edit Distance Matrix**, and test **Morpheme Anatomy** on `'unhappiness'`.
* **Speaker:**
  > *"Next, we compare morphological reduction: **Stemming vs. Lemmatization**.*
  >
  > *Stemmers like Porter and Snowball use heuristic suffix stripping. Notice in our side-by-side table that `studies` is chopped into the pseudo-word `studi`, and `flies` becomes `fli`.*
  >
  > *In contrast, the WordNet Lemmatizer uses vocabulary databases and grammar rules to return true dictionary base words: `study` and `fly`.*
  >
  > *Our app automatically highlights discrepancies in pastel pink, showing students exactly where and why stemmers fail on irregular inflections like `mice` and `feet`.*
  >
  > *Below the table, our **Minimum Edit Distance (Levenshtein Matrix) Visualizer** calculates the dynamic programming alignment grid step-by-step between any two words, showing the exact insertion, deletion, and substitution costs!*
  >
  > *Scroll down to our **Morpheme Anatomy & Affix Inspector**: it breaks complex words like `'unhappiness'` into `[Prefix: un-]` + `[Root: happy]` + `[Suffix: -ness]`, teaching students the critical distinction between derivational and inflectional morphology!"*

---

### 2:15 – 3:00 | Tabs 3 & 4: POS Tagging, Shallow Chunking & DisplaCy NER
* **Action:** Click on **🏷️ POS Tags**, scroll to the **Syntactic Dependency Tree**, expand the **POS Transition Flow Matrix**, showcase the **Syntactic Noun Phrase Chunking** section, then switch to **🔍 NER** to demonstrate the **Entity Category Filter** and **Co-occurrence Links**.
* **Speaker:**
  > *"In **Tab 3**, we use spaCy's statistical tagger to classify every word into its grammatical role (Noun, Verb, Adjective, etc.) with a live Plotly frequency bar chart and human-readable Penn Treebank explanations.*
  >
  > *Scroll down to our live **Syntactic Dependency Parse Tree**, which renders curved grammatical arcs connecting Subject, Root Verb, and Direct Object.*
  >
  > *Expand our **POS Transition Probability Matrix**: it models bigram grammar flow as a Markovian heatmap, showing how likely determiners lead to nouns or verbs lead to adverbs—the core logic behind HMMs and CRFs!*
  >
  > *Directly below, look at our **Syntactic Noun Phrase Chunking (Shallow Parsing)** visualizer: it groups words into non-overlapping grammatical phrases like `'the entire wildlife experiment'`, identifies the root head word (`experiment`), and explains its syntactic function in the sentence.*
  >
  > *Moving to **Tab 4: Named Entity Recognition**, we integrate spaCy's DisplaCy visualizer styled with our light pastel palette. On our benchmark sentence:*
  > `'Elon Musk visited Mumbai on 5 May 2024 to announce Tesla's $2 billion investment...'`
  > *The model extracts `Elon Musk` as a PERSON, `Mumbai` as a GPE, `Tesla` as an ORG, and `$2 billion` as MONEY, complete with start-end character spans.*
  >
  > *Notice our **Entity Category Multiselect Filter** and the **Entity Co-occurrence Links** expander, which automatically extracts entity pairings to seed relation extraction in modern Knowledge Graphs!"*

---

### 3:00 – 3:55 | Tabs 5 & 6: TF-IDF, TextRank Summarizer & Vector Geometry
* **Action:** Click on **📊 TF-IDF**, show the N-gram feature range selector `(1, 2)` for compound phrases, the **Inter-Document Similarity Heatmap**, run a live query in the **Search Engine Simulator**, and scroll to the **TextRank Extractive Summarizer**. Next, switch to **🌌 Embeddings**: show the **Top-K Nearest Neighbors Explorer**, the **Distance Metric Comparison Matrix**, the **Open-Vocabulary Analogy Solver**, toggle to **3D Orbit**, and showcase the **Parallelogram Law Visualizer**.
* **Speaker:**
  > *"In **Tab 5**, we show how computers convert text into numerical weights. Our **N-Gram Range Selector** allows extracting compound terms like `'natural language'` and `'artificial intelligence'` alongside unigrams.*
  >
  > *To answer why TF-IDF was invented, look at our live **Search Engine / Information Retrieval Simulator**: when we query `'artificial intelligence'`, the engine vectorizes our query and calculates cosine similarity across the entire corpus in real time, immediately ranking relevant documents with match percentages!*
  >
  > *Directly below, look at our **Extractive Text Summarization Engine (TextRank)**: it applies Google's PageRank algorithm over an inter-sentence cosine similarity graph, scoring the recursive centrality of each sentence to synthesize a concise, high-information summary with quantifiable compression metrics.*
  >
  > *Finally, in **Tab 6: Word Embeddings**, we extract 300-dimensional dense vectors from spaCy's `en_core_web_md`.*
  >
  > *Notice our **Distance Metric Comparison Matrix**: it computes Cosine Similarity, Cosine Distance, Euclidean ($L_2$), and Manhattan ($L_1$) distances side-by-side, visually explaining why Cosine is preferred in 300D space due to length invariance.*
  >
  > *With our **Top-K Nearest Neighbors Explorer**, users can search any word—like `'hospital'`—and immediately see its closest semantic neighbors like `'clinic'` and `'doctor'` ranked with similarity bars.*
  >
  > *Our **Cluster Cosine Similarity Heatmap** proves that semantic categories (Royalty, Fruits, Vehicles) form distinct high-density blocks in vector space.*
  >
  > *In the 2D projection, our **Semantic Trajectory Arrows** visually link analogical pairs like `Man ➔ King` and `Woman ➔ Queen`. You can also toggle to our **3D Interactive Orbit** to freely rotate and zoom into semantic clusters.*
  >
  > *Even more exciting is our **Open-Vocabulary Semantic Analogy Solver**: it searches through thousands of words in continuous vector space to solve equations like $\text{King} - \text{Man} + \text{Woman} = \text{Queen}$ dynamically!*
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
