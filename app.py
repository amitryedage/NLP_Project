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

# Curated Sample Sentences for Global Testing
SAMPLE_SENTENCES = {
    "🌐 NLP & Language Intro": (
        "Natural Language Processing (NLP) enables computers to understand, interpret, "
        "and manipulate human speech and text. Don't you think it's fascinating? "
        "In 2026, modern AI models can process millions of sentences in seconds!"
    ),
    "🚀 Elon Musk & Industry NER": (
        "Elon Musk visited Mumbai on 5 May 2024 to announce Tesla's $2 billion investment "
        "alongside Indian Prime Minister Narendra Modi and local officials."
    ),
    "🌱 Morphological Studies": (
        "The experienced researchers were studying flying bats, green leaves, "
        "and various connections while running happily on their feet."
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
    st.markdown("<span style='font-size: 0.9rem; color: #4B5563;'>Select a preset to auto-populate test text:</span>", unsafe_allow_html=True)
    p_cols = st.columns(3)
    for idx, (label, sent) in enumerate(SAMPLE_SENTENCES.items()):
        with p_cols[idx]:
            if st.button(f"{label}", use_container_width=True, key=f"btn_preset_{idx}"):
                st.session_state["shared_input"] = sent
                st.rerun()

    if "shared_input" in st.session_state:
        st.info(f"**Active Preset Loaded:** \"{st.session_state['shared_input'][:90]}...\"")
        if st.button("Reset to Defaults", key="reset_preset"):
            del st.session_state["shared_input"]
            st.rerun()

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
