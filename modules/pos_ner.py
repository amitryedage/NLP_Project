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

    # 8. Syntactic Dependency Parse Tree (DisplaCy)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌳 Syntactic Dependency Parse Tree")
    st.caption("Visualizes grammatical dependencies and relationships (Subject → Root Verb → Direct Object):")

    dep_options = {
        "compact": True,
        "color": COLOR_TEXT,
        "bg": COLOR_SECONDARY_BG,
        "font": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
        "distance": 110,
    }
    try:
        dep_svg = displacy.render(doc, style="dep", options=dep_options, page=False)
        st.markdown(
            f"""
            <div style="overflow-x: auto; background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; 
                        border-radius: 12px; padding: 18px; margin-bottom: 20px;">
                {dep_svg}
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.info(f"Dependency tree visualizer ready for standard sentences.")

    # 8b. POS Transition Probability Matrix (Markov Grammar Flow)
    with st.expander("🔄 POS Transition Probability Matrix (Markov Grammar Flow)", expanded=False):
        st.markdown(
            f"<span style='color: {COLOR_MUTED}; font-size: 0.9rem;'>"
            f"Shows the probability $P(\\text{{Tag}}_t \\mid \\text{{Tag}}_{{t-1}})$ of one part of speech following another. "
            f"This Markovian transition matrix forms the statistical backbone of sequence taggers (HMMs & CRFs):"
            f"</span>",
            unsafe_allow_html=True,
        )

        pos_sequence = [token.pos_ for token in doc if not token.is_space]
        if len(pos_sequence) >= 2:
            unique_pos_sorted = sorted(list(set(pos_sequence)))
            trans_counts = pd.DataFrame(0, index=unique_pos_sorted, columns=unique_pos_sorted)

            for i in range(len(pos_sequence) - 1):
                prev_tag = pos_sequence[i]
                next_tag = pos_sequence[i + 1]
                trans_counts.loc[prev_tag, next_tag] += 1

            # Normalize row-wise to get transition probabilities P(next | prev)
            row_sums = trans_counts.sum(axis=1)
            trans_probs = trans_counts.div(row_sums.replace(0, 1), axis=0) * 100

            fig_trans = px.imshow(
                trans_probs,
                labels=dict(x="Next Tag (t)", y="Current Tag (t-1)", color="Transition Probability (%)"),
                x=unique_pos_sorted,
                y=unique_pos_sorted,
                color_continuous_scale="Blues",
                text_auto=".0f",
                aspect="auto",
                template="plotly_white",
            )
            fig_trans.update_layout(
                title="POS-to-POS Transition Probability Matrix (%)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor=COLOR_BG,
                paper_bgcolor=COLOR_BG,
            )
            st.plotly_chart(fig_trans, use_container_width=True)

            # Highlight dominant grammatical patterns detected
            st.markdown("<b>Frequent Grammar Patterns Detected in Input:</b>", unsafe_allow_html=True)
            pattern_chips = []
            for i in range(len(pos_sequence) - 1):
                p_tag, n_tag = pos_sequence[i], pos_sequence[i + 1]
                p_word, n_word = doc[i].text, doc[i + 1].text
                if (p_tag, n_tag) in [("DET", "NOUN"), ("DET", "ADJ"), ("ADJ", "NOUN"), ("NOUN", "VERB"), ("VERB", "ADV"), ("ADP", "DET"), ("VERB", "NOUN")]:
                    pattern_label = f"{p_word} ({p_tag}) ➔ {n_word} ({n_tag})"
                    pattern_chips.append(chip(pattern_label, color=PASTEL_BLUE))
            if pattern_chips:
                seen_pat = set()
                dedup_chips = []
                for pc in pattern_chips:
                    if pc not in seen_pat:
                        seen_pat.add(pc)
                        dedup_chips.append(pc)
                st.markdown(f'<div class="chip-container">{" ".join(dedup_chips[:8])}</div>', unsafe_allow_html=True)
            else:
                st.caption("Standard syntactic sequences detected.")
        else:
            st.info("Enter at least 2 tokens to compute POS transition probabilities.")

    # 9. Detailed Tabular Breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Grammatical Breakdown Table")
    df_table = pd.DataFrame(token_records)[["Index", "Word", "Universal POS", "Fine Tag", "Grammatical Meaning", "Lemma"]]
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Export POS Table Button
    st.download_button(
        label="📥 Download POS Breakdown (CSV)",
        data=df_table.to_csv(index=False).encode("utf-8"),
        file_name="pos_tagging_breakdown.csv",
        mime="text/csv",
        key="btn_download_pos_table"
    )

    # 9b. Syntactic Noun Phrase Chunking (Shallow Parsing)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📦 Syntactic Noun Phrase Chunking (Shallow Parsing)")
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: {COLOR_TEXT}; margin-bottom: 3px;">
                What is Shallow Parsing (Chunking)?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.88rem; line-height: 1.4;">
                While full dependency parsing constructs an entire tree, <b>Chunking</b> extracts non-overlapping, 
                syntactically correlated word phrases. In English, <b>Noun Chunks</b> (base noun phrases) capture 
                core conceptual entities along with their descriptive determiners and adjectives:
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    chunks = list(doc.noun_chunks)
    if chunks:
        chunk_records = []
        chunk_chips = []
        for idx, chunk in enumerate(chunks):
            color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
            head_word = chunk.root.text
            dep_role = chunk.root.dep_
            dep_explain = spacy.explain(dep_role) or dep_role
            chunk_chips.append(chip(chunk.text, color=color, subtitle=f"Head: {head_word} ({dep_role})"))
            chunk_records.append({
                "Phrase Index": idx + 1,
                "Noun Chunk Phrase": chunk.text,
                "Root Head Word": head_word,
                "Root POS": chunk.root.pos_,
                "Syntactic Dependency": dep_role,
                "Grammatical Function": dep_explain,
                "Token Span": f"{chunk.start} to {chunk.end}",
                "Length (Tokens)": len(chunk)
            })

        st.markdown("**Detected Noun Chunks (Grouped Syntactic Units):**")
        st.markdown(f'<div class="chip-container">{" ".join(chunk_chips)}</div>', unsafe_allow_html=True)

        df_chunks = pd.DataFrame(chunk_records)
        c_ch1, c_ch2 = st.columns([1, 2])
        with c_ch1:
            st.metric("Total Noun Chunks", len(chunks))
            avg_chunk_len = df_chunks["Length (Tokens)"].mean()
            st.metric("Avg Chunk Length", f"{avg_chunk_len:.1f} tokens")
            longest_chunk = df_chunks.loc[df_chunks["Length (Tokens)"].idxmax()]["Noun Chunk Phrase"]
            st.metric("Longest Phrase", f'"{longest_chunk[:18]}..."' if len(longest_chunk) > 18 else f'"{longest_chunk}"')

        with c_ch2:
            fig_chunks = px.bar(
                df_chunks,
                x="Noun Chunk Phrase",
                y="Length (Tokens)",
                text="Root Head Word",
                title="Noun Phrase Token Length & Syntactic Heads",
                color="Length (Tokens)",
                color_continuous_scale="Purples"
            )
            fig_chunks.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=35, b=20),
                plot_bgcolor=COLOR_BG,
                paper_bgcolor=COLOR_BG
            )
            st.plotly_chart(fig_chunks, use_container_width=True)

        st.dataframe(df_chunks[["Phrase Index", "Noun Chunk Phrase", "Root Head Word", "Syntactic Dependency", "Grammatical Function", "Token Span"]], use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Download Noun Chunks (CSV)",
            data=df_chunks.to_csv(index=False).encode("utf-8"),
            file_name="noun_chunks_breakdown.csv",
            mime="text/csv",
            key="btn_download_noun_chunks"
        )
    else:
        st.info("No base noun chunks identified in the active input sentence.")

    # 10. Educational Insights Box
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

    # 7. Entity Category Filter
    col_filter, _ = st.columns([2, 1])
    with col_filter:
        selected_types = st.multiselect(
            "Filter Displayed Entities by Category:",
            options=sorted(list(entity_counts.keys())),
            default=sorted(list(entity_counts.keys())),
            key="ner_category_multiselect"
        )

    filtered_records = [r for r in entity_records if r["Label / Type"] in selected_types]

    st.subheader("📌 Extracted Entities")
    if filtered_records:
        grouped_chips = []
        for rec in filtered_records:
            color = rec["Color"]
            grouped_chips.append(chip(text=rec["Entity Text"], color=color, subtitle=rec["Label / Type"]))
        st.markdown(f'<div class="chip-container">{" ".join(grouped_chips)}</div>', unsafe_allow_html=True)
    else:
        st.info("No entities match the selected category filter.")

    # 8. Structured Entity Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Entity Details Table")
    df_entities = pd.DataFrame(filtered_records)[["Entity Text", "Label / Type", "Category Meaning", "Start Char", "End Char"]] if filtered_records else pd.DataFrame(columns=["Entity Text", "Label / Type", "Category Meaning", "Start Char", "End Char"])
    st.dataframe(df_entities, use_container_width=True, hide_index=True)

    # Export NER Entities Button
    st.download_button(
        label="📥 Download Entities Table (CSV)",
        data=df_entities.to_csv(index=False).encode("utf-8"),
        file_name="ner_entities.csv",
        mime="text/csv",
        key="btn_download_ner_table"
    )

    # 8b. Entity Co-occurrence & Relationship Graph
    if len(entities) >= 2:
        with st.expander("🌐 Entity Co-occurrence & Relationship Links", expanded=False):
            st.markdown(
                f"<span style='color: {COLOR_MUTED}; font-size: 0.9rem;'>"
                f"Extracts co-occurring entities that appear together in the text to seed Knowledge Graphs and relation extraction:"
                f"</span>",
                unsafe_allow_html=True,
            )
            rel_chips = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = entities[i], entities[j]
                    rel_label = f"{e1.text} ({e1.label_}) ⟷ {e2.text} ({e2.label_})"
                    rel_chips.append(chip(rel_label, color=PASTEL_PURPLE))
            st.markdown(f'<div class="chip-container">{" ".join(rel_chips[:10])}</div>', unsafe_allow_html=True)

    # 9. Educational Takeaway Box
    with st.expander("💡 Deep Dive: How spaCy Identifies Named Entities"):
        st.markdown(
            """
            * **GPE vs. LOC:** `GPE` (Geopolitical Entity) refers to countries, cities, and states with governing bodies (e.g., *Mumbai*, *India*, *France*). `LOC` refers to non-GPE physical locations (e.g., *Himalayas*, *Pacific Ocean*).
            * **Context is Key:** If a sentence says *"Apple launched a new iPhone"*, the model labels `Apple` as an `ORG`. But in *"I ate a fresh red apple"*, the model recognizes it is a common noun and ignores it.
            * **Statistical Sequence Tagging:** spaCy uses transition-based neural networks (with Word2Vec / GloVe embeddings) to detect both entity boundaries and categories simultaneously.
            """
        )
