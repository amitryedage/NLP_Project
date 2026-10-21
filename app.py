"""
NLP Visual Lab - Main Streamlit Application
Interactive educational visualizer for core NLP concepts.
Department of AI & Data Science, ADCET Ashta
"""
import streamlit as st
from utils.theme import (
    CUSTOM_CSS,
    COLOR_PRIMARY,
    COLOR_SECONDARY_BG,
    COLOR_BORDER,
    COLOR_TEXT,
    COLOR_MUTED,
)
from modules.tokenization import render_tokenization_tab
from modules.stemming_lemma import render_stemming_lemma_tab
from modules.pos_ner import render_pos_tab, render_ner_tab
from modules.tfidf import render_tfidf_tab
from modules.embeddings import render_embeddings_tab

# Page Configuration - Strictly Light Theme
st.set_page_config(
    page_title="NLP Visual Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Light Theme Styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Curated Sample Sentences for Global Testing across Diverse Domains
SAMPLE_SENTENCES = {
    "🌐 NLP Intro": (
        "Natural Language Processing (NLP) enables computers to understand, interpret, "
        "and manipulate human speech and text. Don't you think it's fascinating? "
        "In 2026, modern AI models can process millions of sentences in seconds!"
    ),
    "🚀 Industry & NER": (
        "Elon Musk visited Mumbai on 5 May 2024 to announce Tesla's $2 billion investment "
        "alongside Indian Prime Minister Narendra Modi and local officials."
    ),
    "🌱 Morphology": (
        "The experienced researchers were studying flying bats, green leaves, "
        "and various connections while running happily on their feet."
    ),
    "🔬 Healthcare": (
        "The clinical oncologist administered antibody immunotherapy to leukemia patients, "
        "observing significant tumor regression during pharmaceutical trials in Boston."
    ),
    "🐦 Social Tweet": (
        "Loving the new #GenerativeAI breakthroughs in 2026! Can't wait to test Gemini & spaCy models. "
        "Are you excited? 🔥✨ #NLP #MachineLearning"
    ),
}

# --- App Header ---
col_head, col_badge = st.columns([4, 1])
with col_head:
    st.title("🔬 NLP Visual Lab")
    st.markdown(
        "An interactive educational visualizer to see how core **Natural Language Processing** "
        "algorithms transform text step-by-step. Live computation on your own inputs — no static mockups."
    )
with col_badge:
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; 
                    border-radius: 10px; padding: 10px; text-align: center; margin-top: 10px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: {COLOR_PRIMARY}; text-transform: uppercase;">Course Micro Project</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: {COLOR_TEXT};">NLP (1ADPC402)</div>
            <div style="font-size: 0.72rem; color: {COLOR_MUTED};">ADCET Ashta (AI & DS)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Try an Example Toolbar ---
with st.expander("⚡ Try Quick Preset Sentences (Click to load across tabs)", expanded=False):
    st.markdown("<span style='font-size: 0.9rem; color: #4B5563;'>Select a preset domain to auto-populate test text across Tabs 1 to 4:</span>", unsafe_allow_html=True)
    p_cols = st.columns(len(SAMPLE_SENTENCES))
    for idx, (label, sent) in enumerate(SAMPLE_SENTENCES.items()):
        with p_cols[idx]:
            if st.button(f"{label}", use_container_width=True, key=f"btn_preset_{idx}"):
                st.session_state["shared_input"] = sent
                st.rerun()

    if "shared_input" in st.session_state:
        st.info(f"**Active Preset Loaded:** \"{st.session_state['shared_input'][:95]}...\"")
        if st.button("Reset to Defaults", key="reset_preset"):
            del st.session_state["shared_input"]
            st.rerun()

# --- Live Linguistic Readability & Text Statistics Strip ---
active_sample = st.session_state.get("shared_input", SAMPLE_SENTENCES["🌐 NLP Intro"])

def compute_readability(text: str):
    import re
    from collections import Counter
    words = [w for w in re.findall(r'\b\w+\b', text) if any(c.isalpha() for c in w)]
    w_count = len(words)
    s_count = max(1, len([s for s in re.split(r'[.!?]+', text.strip()) if s.strip()]))
    c_count = sum(len(w) for w in words)
    words_lower = [w.lower() for w in words]
    vocab_size = len(set(words_lower))
    word_freqs = Counter(words_lower)
    hapax_count = sum(1 for w, c in word_freqs.items() if c == 1)
    ttr = (vocab_size / w_count * 100) if w_count > 0 else 0.0

    def count_syllables(w):
        w = w.lower()
        if len(w) <= 3:
            return 1
        w = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', w)
        w = re.sub(r'^y', '', w)
        matches = re.findall(r'[aeiouy]{1,2}', w)
        return max(1, len(matches))

    syl_count = sum(count_syllables(w) for w in words)

    if w_count > 0:
        asl = w_count / s_count
        asw = syl_count / w_count
        flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
        flesch = max(0.0, min(100.0, flesch))
    else:
        flesch = 70.0

    return {
        "words": w_count,
        "sentences": s_count,
        "chars": c_count,
        "vocab": vocab_size,
        "ttr": ttr,
        "hapax": hapax_count,
        "flesch": flesch,
    }

read_stats = compute_readability(active_sample)

with st.container():
    c_st1, c_st2, c_st3, c_st4, c_st5 = st.columns([1, 1, 1, 1, 1.2])
    with c_st1:
        st.metric("Words & Sentences", f"{read_stats['words']}w · {read_stats['sentences']}s")
    with c_st2:
        st.metric("Unique Vocab (|V|)", f"{read_stats['vocab']} terms")
    with c_st3:
        st.metric(
            "Type-Token Ratio",
            f"{read_stats['ttr']:.1f}%",
            help="TTR = (|V| / N) * 100. Measures lexical diversity and repetition in text."
        )
    with c_st4:
        st.metric(
            "Hapax Legomena",
            f"{read_stats['hapax']} words",
            help="Terms occurring exactly once in the input text, measuring linguistic novelty."
        )
    with c_st5:
        st.metric(
            "Flesch Reading Ease",
            f"{read_stats['flesch']:.1f}/100",
            help="Higher scores (60-100) indicate accessible English; lower scores indicate dense scholarly syntax."
        )

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# --- Sidebar Controls & Documentation ---
with st.sidebar:
    st.header("🔬 NLP Visual Lab")
    st.markdown(
        """
        **Interactive Pipeline Explorer**
        
        Navigate through the tabs to inspect each step in modern NLP:
        
        1. **✂️ Tokenization:** Word, Sentence & spaCy
        2. **🌱 Stem vs. Lemma:** Heuristic vs. Lexical
        3. **🏷️ POS Tags:** Grammatical category tags
        4. **🔍 NER:** Named entity extraction
        5. **📊 TF-IDF:** Bag of Words & weighting
        6. **🌌 Embeddings:** 300D PCA vector space
        """
    )
    st.divider()
    st.markdown("### 🎓 Academic Information")
    st.markdown(
        """
        - **Course:** Natural Language Processing (1ADPC402)
        - **Department:** Artificial Intelligence & Data Science
        - **Institution:** Annasaheb Dange College of Engineering and Technology (ADCET), Ashta
        - **Stack:** Python 3.12, Astral `uv`, Streamlit, spaCy, NLTK, scikit-learn, Plotly
        - **Theme:** Strict Light Theme
        """
    )
    st.divider()
    st.caption("Developed for high-clarity NLP visual comprehension.")

# --- Tab Definitions (6 Concept Tabs) ---
tab_names = [
    "✂️ Tokenization",
    "🌱 Stem / Lemma",
    "🏷️ POS Tags",
    "🔍 NER",
    "📊 TF-IDF",
    "🌌 Embeddings",
]

tab_tokens, tab_stems, tab_pos, tab_ner, tab_tfidf, tab_embeddings = st.tabs(tab_names)

with tab_tokens:
    render_tokenization_tab()

with tab_stems:
    render_stemming_lemma_tab()

with tab_pos:
    render_pos_tab()

with tab_ner:
    render_ner_tab()

with tab_tfidf:
    render_tfidf_tab()

with tab_embeddings:
    render_embeddings_tab()

# --- Page Footer ---
st.markdown("<br><hr style='border: 0; border-top: 1px solid #E5E7EB; margin: 30px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; color: {COLOR_MUTED}; font-size: 0.85rem; padding-bottom: 24px;">
        NLP Visual Lab • Natural Language Processing (1ADPC402) • ADCET Ashta (AI & Data Science)
    </div>
    """,
    unsafe_allow_html=True,
)
