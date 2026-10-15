"""
Module for Part-of-Speech (POS) Tagging and Named Entity Recognition (NER).
Implements Tabs 3 and 4 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
from spacy import displacy
from utils.theme import (
    POS_COLOR_MAP,
    ENTITY_COLOR_MAP,
    PASTEL_BLUE,
    PASTEL_COLORS,
    chip,
    COLOR_BG,
    COLOR_SECONDARY_BG,
    COLOR_BORDER,
    COLOR_TEXT,
    COLOR_MUTED,
    COLOR_PRIMARY,
)
from utils.nlp_loader import load_spacy_model

# Default sample texts
DEFAULT_POS_TEXT = (
    "The quick brown fox jumps gracefully over a lazy dog, while curious researchers "
    "observe the entire wildlife experiment in Mumbai."
)

DEFAULT_NER_TEXT = (
    "Elon Musk visited Mumbai on 5 May 2024 to announce Tesla's $2 billion investment "
    "alongside Indian Prime Minister Narendra Modi and local officials."
)


# =====================================================================
# TAB 3: Part-of-Speech (POS) Tagging
# =====================================================================
def render_pos_tab():
    """
    Renders the interactive POS Tagging visualizer with pastel tags,
    color legends, frequency charts, and Penn Treebank explanations.
    """
    # 1. Plain-English Concept Explanation
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1.05rem; color: {COLOR_TEXT}; margin-bottom: 4px;">
                🏷️ What is Part-of-Speech (POS) Tagging?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.93rem; line-height: 1.5;">
                Part-of-Speech (POS) Tagging assigns grammatical labels to each word in a sentence 
                (such as <b>Noun</b>, <b>Verb</b>, <b>Adjective</b>, or <b>Adverb</b>) based on its definition 
                and surrounding context. This enables NLP models to distinguish word syntax and grammatical roles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. User Input Area
    default_val = st.session_state.get("shared_input", DEFAULT_POS_TEXT)
    user_text = st.text_area(
        label="Enter sentence for POS tagging:",
        value=default_val,
        height=95,
        max_chars=1000,
        help="Type or paste any text to see real-time grammatical tagging.",
        key="pos_text_input",
    )

    # Edge Case: Empty input
    if not user_text or not user_text.strip():
        st.info("👆 Please enter a sentence above to view its grammatical parts of speech.")
        return

    # 3. Model Execution
    nlp = load_spacy_model()
    doc = nlp(user_text)

    if len(doc) == 0:
        st.warning("No tokens found in the provided input.")
        return

    # 4. Process Tokens & Tags
    token_records = []
    pos_counts = {}
    present_pos_types = set()

    for idx, token in enumerate(doc):
        pos_tag = token.pos_
        fine_tag = token.tag_
        
        # Human readable explanation via spacy.explain
        try:
            explanation = spacy.explain(fine_tag) or spacy.explain(pos_tag) or "Punctuation or Symbol"
        except Exception:
            explanation = pos_tag

        color = POS_COLOR_MAP.get(pos_tag, "#E5E7EB")
        present_pos_types.add(pos_tag)
        pos_counts[pos_tag] = pos_counts.get(pos_tag, 0) + 1

        token_records.append({
            "Index": idx + 1,
            "Word": token.text,
            "Universal POS": pos_tag,
            "Fine Tag": fine_tag,
            "Grammatical Meaning": explanation.capitalize(),
            "Lemma": token.lemma_,
            "Color": color,
        })

    # 5. Interactive Color Legend
    st.subheader("🎨 POS Color Legend")
    legend_chips = []
    for pos_type in sorted(present_pos_types):
        color = POS_COLOR_MAP.get(pos_type, "#E5E7EB")
        meaning = spacy.explain(pos_type) or pos_type
        legend_chips.append(chip(f"{pos_type}: {meaning.title()}", color=color))

    st.markdown(
        f'<div class="chip-container">{" ".join(legend_chips)}</div>',
        unsafe_allow_html=True,
    )

    # 6. Live Color-Coded Word Stream
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔍 Tagged Word Stream")
    st.caption("Each word is highlighted by its grammatical role with its universal POS sub-label:")

    chips_html = []
    for rec in token_records:
        chips_html.append(chip(text=rec["Word"], color=rec["Color"], subtitle=rec["Universal POS"]))

    st.markdown(f'<div class="chip-container">{" ".join(chips_html)}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 7. Metrics & POS Distribution Chart
    col_chart, col_metrics = st.columns([3, 2])

    with col_chart:
        st.subheader("📊 POS Frequency Distribution")
        df_counts = pd.DataFrame(
            list(pos_counts.items()),
            columns=["POS Category", "Count"]
        ).sort_values(by="Count", ascending=False)

        # Plotly chart with light theme
        colors_for_bars = [POS_COLOR_MAP.get(p, "#4F46E5") for p in df_counts["POS Category"]]
        fig = px.bar(
            df_counts,
            x="POS Category",
            y="Count",
            text="Count",
            color="POS Category",
            color_discrete_map=POS_COLOR_MAP,
            template="plotly_white",
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            xaxis_title=None,
            yaxis_title="Frequency",
            plot_bgcolor=COLOR_BG,
            paper_bgcolor=COLOR_BG,
        )
        fig.update_traces(
            textposition="outside",
            marker=dict(line=dict(width=1, color=COLOR_BORDER)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.subheader("📈 Sentence Statistics")
        st.metric("Total Tokens Tagged", len(token_records))
        st.metric("Distinct POS Categories", len(present_pos_types))
        most_common_pos = max(pos_counts, key=pos_counts.get)
        st.metric("Most Frequent POS", f"{most_common_pos} ({pos_counts[most_common_pos]}x)")

    # 8. Detailed Tabular Breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Grammatical Breakdown Table")
    df_table = pd.DataFrame(token_records)[["Index", "Word", "Universal POS", "Fine Tag", "Grammatical Meaning", "Lemma"]]
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # 9. Educational Insights Box
    with st.expander("💡 Deep Dive: Contextual Ambiguity in POS Tagging"):
        st.markdown(
            """
            * **Lexical Ambiguity:** Many English words change their part of speech based entirely on context:
              * *"Time **flies** like an arrow."* → `flies` is a **VERB** (`VBZ`).
              * *"Fruit **flies** like a banana."* → `flies` is a **NOUN** (`NNS`).
              * *"Can you **book** a flight?"* → `book` is a **VERB**.
              * *"I read a good **book**."* → `book` is a **NOUN**.
            * spaCy uses a statistical contextual tagger (transition-based parsing) that looks at surrounding tokens to resolve these ambiguities accurately.
            """
        )


# =====================================================================
# TAB 4: Named Entity Recognition (NER)
# =====================================================================
def render_ner_tab():
    """
    Renders the Named Entity Recognition visualizer with DisplaCy integration,
    human-readable entity definitions, and extracted entity tables.
    """
    # 1. Plain-English Concept Explanation
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1.05rem; color: {COLOR_TEXT}; margin-bottom: 4px;">
                🔍 What is Named Entity Recognition (NER)?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.93rem; line-height: 1.5;">
                Named Entity Recognition (NER) automatically scans raw text to identify and classify real-world 
                entities into structured categories such as <b>People</b>, <b>Organizations</b>, <b>Locations</b>, 
                <b>Dates</b>, and <b>Monetary Values</b>.
                <br>
                <b>Real-World Applications:</b> Search engines, automated resume filtering, news aggregation, 
                financial risk detection, and customer support ticket routing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. User Input Area
    default_val = st.session_state.get("shared_input", DEFAULT_NER_TEXT)
    user_text = st.text_area(
        label="Enter text for Named Entity Recognition:",
        value=default_val,
        height=95,
        max_chars=1000,
        help="Type or paste any text mentioning people, places, dates, or organizations.",
        key="ner_text_input",
    )

    # Edge Case: Empty input
    if not user_text or not user_text.strip():
        st.info("👆 Please enter text above to extract named entities.")
        return

    # 3. Model Execution
    nlp = load_spacy_model()
    doc = nlp(user_text)

    # 4. DisplaCy Visualizer with Custom Pastel Light Theme
    st.subheader("🏷️ Interactive DisplaCy Entity Visualizer")
    st.caption("Entities are highlighted inline using light pastel tones with clear category tags:")

    displacy_colors = {
        "PERSON": "#DBEAFE",      # Pastel Blue
        "ORG": "#EDE9FE",         # Pastel Purple
        "GPE": "#DCFCE7",         # Pastel Green
        "LOC": "#DCFCE7",         # Pastel Green
        "DATE": "#FEF9C3",        # Pastel Yellow
        "TIME": "#FEF9C3",        # Pastel Yellow
        "MONEY": "#FFEDD5",       # Pastel Orange
        "PERCENT": "#FFEDD5",     # Pastel Orange
        "CARDINAL": "#FCE7F3",    # Pastel Pink
        "ORDINAL": "#FCE7F3",     # Pastel Pink
        "NORP": "#FEF9C3",        # Pastel Yellow
        "EVENT": "#EDE9FE",       # Pastel Purple
        "WORK_OF_ART": "#DBEAFE", # Pastel Blue
        "PRODUCT": "#FFEDD5",     # Pastel Orange
    }

    displacy_options = {
        "colors": displacy_colors,
    }

    # Render DisplaCy HTML
    displacy_html = displacy.render(doc, style="ent", options=displacy_options, page=False)
    
    # Wrap in custom styled card ensuring dark text (#1F2937) and light background
    wrapped_html = f"""
    <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; 
                border-radius: 12px; padding: 20px; font-size: 1.05rem; line-height: 2.2; 
                color: {COLOR_TEXT}; margin-bottom: 20px;">
        {displacy_html}
    </div>
    """
    st.markdown(wrapped_html, unsafe_allow_html=True)

    # 5. Extract Entities & Human-Readable Explanations
    entities = doc.ents

    if not entities:
        st.warning("No named entities were detected in the provided text. Try adding names, countries, dates, or organizations.")
        return

    entity_records = []
    entity_counts = {}

    for idx, ent in enumerate(entities):
        label = ent.label_
        entity_counts[label] = entity_counts.get(label, 0) + 1
        
        # Human-readable explanation using spacy.explain
        try:
            explanation = spacy.explain(label) or "Named entity"
        except Exception:
            explanation = label

        entity_records.append({
            "Entity Text": ent.text,
            "Label / Type": label,
            "Category Meaning": explanation.capitalize(),
            "Start Char": ent.start_char,
            "End Char": ent.end_char,
            "Color": ENTITY_COLOR_MAP.get(label, "#DBEAFE"),
        })

    # 6. Entity Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Entities Found", len(entities))
    with m2:
        st.metric("Entity Categories", len(entity_counts))
    with m3:
        top_category = max(entity_counts, key=entity_counts.get)
        st.metric("Most Prevalent Type", f"{top_category} ({entity_counts[top_category]}x)")

    st.markdown("---")

    # 7. Grouped Entity Chips
    st.subheader("📌 Extracted Entities by Category")
    grouped_chips = []
    for rec in entity_records:
        color = rec["Color"]
        grouped_chips.append(chip(text=rec["Entity Text"], color=color, subtitle=rec["Label / Type"]))

    st.markdown(f'<div class="chip-container">{" ".join(grouped_chips)}</div>', unsafe_allow_html=True)

    # 8. Structured Entity Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Entity Details Table")
    df_entities = pd.DataFrame(entity_records)[["Entity Text", "Label / Type", "Category Meaning", "Start Char", "End Char"]]
    st.dataframe(df_entities, use_container_width=True, hide_index=True)

    # 9. Educational Takeaway Box
    with st.expander("💡 Deep Dive: How spaCy Identifies Named Entities"):
        st.markdown(
            """
            * **GPE vs. LOC:** `GPE` (Geopolitical Entity) refers to countries, cities, and states with governing bodies (e.g., *Mumbai*, *India*, *France*). `LOC` refers to non-GPE physical locations (e.g., *Himalayas*, *Pacific Ocean*).
            * **Context is Key:** If a sentence says *"Apple launched a new iPhone"*, the model labels `Apple` as an `ORG`. But in *"I ate a fresh red apple"*, the model recognizes it is a common noun and ignores it.
            * **Statistical Sequence Tagging:** spaCy uses transition-based neural networks (with Word2Vec / GloVe embeddings) to detect both entity boundaries and categories simultaneously.
            """
        )
