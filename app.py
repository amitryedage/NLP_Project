"""
NLP Visual Lab - Main Streamlit Application
Interactive educational visualizer for core NLP concepts.
"""
import streamlit as st
from utils.theme import CUSTOM_CSS, COLOR_PRIMARY
from modules.tokenization import render_tokenization_tab
from modules.stemming_lemma import render_stemming_lemma_tab
from modules.pos_ner import render_pos_tab, render_ner_tab
from modules.tfidf import render_tfidf_tab
from modules.embeddings import render_embeddings_tab

# Page Configuration - Light Theme
st.set_page_config(
    page_title="NLP Visual Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Light Theme Styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# App Header
st.title("🔬 NLP Visual Lab")
st.markdown(
    "Welcome to the **NLP Visual Lab**! An interactive playground where you can "
    "type any sentence and *see* how fundamental Natural Language Processing algorithms work step-by-step."
)

# Sidebar Information
with st.sidebar:
    st.header("About NLP Visual Lab")
    st.markdown(
        """
        - **Course:** NLP Micro Project (1ADPC402)
        - **Department:** AI & Data Science, ADCET Ashta
        - **Theme:** Light
        - **Stack:** Python, Streamlit, spaCy, NLTK, scikit-learn, Plotly
        """
    )
    st.divider()
    st.caption("Designed for intuitive, visual understanding of NLP pipelines.")

# Tab definitions (6 Concept Tabs as required in PRD)
tab_names = [
    "✂️ Tokenization",
    "🌱 Stem / Lemma",
    "🏷️ POS Tags",
    "🔍 NER",
    "📊 TF-IDF",
    "🌌 Embeddings"
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
