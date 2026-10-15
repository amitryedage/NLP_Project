"""
Module for Bag of Words and TF-IDF Vectorization.
Implements Tab 5 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.theme import (
    COLOR_BG,
    COLOR_SECONDARY_BG,
    COLOR_BORDER,
    COLOR_TEXT,
    COLOR_MUTED,
    COLOR_PRIMARY,
    chip,
    PASTEL_BLUE,
    PASTEL_PURPLE,
)

# Default corpus: 4 distinct documents about AI, NLP, and Robotics
DEFAULT_CORPUS = """Natural language processing helps computers understand human language.
Deep learning and neural networks power modern language models.
Computer vision and natural language processing are branches of artificial intelligence.
Artificial intelligence transforms robotics, medicine, and language understanding."""


def render_tfidf_tab():
    """
    Renders the TF-IDF and Bag of Words matrix, heatmap, and keyword analyzer.
    """
    # 1. Plain-English Concept Explanation
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1.05rem; color: {COLOR_TEXT}; margin-bottom: 4px;">
                📊 What is TF-IDF & Bag of Words?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.93rem; line-height: 1.5;">
                Computers cannot calculate text directly; they require numbers. 
                <br>
                • <b>Bag of Words (BoW):</b> Counts raw word occurrences in each document, ignoring order.
                <br>
                • <b>TF-IDF (Term Frequency - Inverse Document Frequency):</b> Measures how important a word is. 
                  Words that appear frequently in <i>one</i> document but rarely across the <i>entire corpus</i> receive the highest scores, 
                  automatically filtering out generic filler words.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. User Input Area (Documents one per line)
    st.subheader("📝 Enter Corpus (3 to 5 Documents, one per line)")
    user_corpus = st.text_area(
        label="Enter documents (one sentence per line):",
        value=DEFAULT_CORPUS,
        height=130,
        help="Enter 3 to 5 sentences. Each line represents a separate document in the corpus.",
        key="tfidf_corpus_input",
    )

    # Split lines and filter empty rows
    doc_lines = [line.strip() for line in user_corpus.splitlines() if line.strip()]

    # Edge Case: Fewer than 2 documents
    if len(doc_lines) < 2:
        st.warning(
            "⚠️ Please enter **at least 2 non-empty document lines** to compute a meaningful TF-IDF matrix. "
            "TF-IDF requires multiple documents to calculate Inverse Document Frequency (IDF)."
        )
        return

    # 3. Vectorization Controls
    col_mode, col_options = st.columns([2, 3])
    with col_mode:
        vectorizer_mode = st.radio(
            "Select Representation Model:",
            ["TF-IDF (Term Frequency - Inverse Document Frequency)", "Bag of Words (CountVectorizer)"],
            index=0,
            horizontal=False,
        )

    with col_options:
        remove_stopwords = st.checkbox(
            "Remove English Stopwords (e.g. 'and', 'are', 'of')",
            value=False,
            help="Stripping common grammatical stopwords prevents them from dominating word frequencies.",
        )
        lowercase = st.checkbox("Convert to Lowercase", value=True)

    st.markdown("---")

    # 4. Scikit-Learn Vectorization
    stop_words_arg = "english" if remove_stopwords else None

    if "TF-IDF" in vectorizer_mode:
        vectorizer = TfidfVectorizer(
            stop_words=stop_words_arg,
            lowercase=lowercase,
            smooth_idf=True,
        )
        matrix_sparse = vectorizer.fit_transform(doc_lines)
        score_label = "TF-IDF Score"
        decimals = 3
    else:
        vectorizer = CountVectorizer(
            stop_words=stop_words_arg,
            lowercase=lowercase,
        )
        matrix_sparse = vectorizer.fit_transform(doc_lines)
        score_label = "Word Count"
        decimals = 0

    feature_names = vectorizer.get_feature_names_out()
    doc_names = [f"Doc {i+1}" for i in range(len(doc_lines))]

    # Build DataFrame
    matrix_dense = matrix_sparse.toarray()
    df_matrix = pd.DataFrame(matrix_dense, index=doc_names, columns=feature_names)

    # 5. Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Documents", len(doc_lines))
    with m2:
        st.metric("Vocabulary Size", len(feature_names))
    with m3:
        sparsity = (df_matrix.values == 0).mean() * 100
        st.metric("Matrix Sparsity", f"{sparsity:.1f}%")
    with m4:
        max_score = df_matrix.values.max()
        st.metric(f"Peak {score_label}", f"{max_score:.2f}" if decimals == 3 else f"{int(max_score)}")

    # 6. Resulting Matrix Table
    st.subheader(f"🔢 {vectorizer_mode.split('(')[0].strip()} Feature Matrix")
    st.caption("Rows represent documents; columns represent unique vocabulary tokens:")

    if decimals == 3:
        formatted_df = df_matrix.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None)
    else:
        formatted_df = df_matrix.style.format("{:d}").background_gradient(cmap="Blues", axis=None)

    st.dataframe(formatted_df, use_container_width=True)

    # 7. Interactive Heatmap (Plotly)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔥 Document-Word Importance Heatmap")
    st.caption("Visualizes the weights across the entire vocabulary using a light color palette:")

    fig_heat = px.imshow(
        df_matrix,
        labels=dict(x="Vocabulary Word", y="Document", color=score_label),
        x=feature_names,
        y=doc_names,
        color_continuous_scale="Blues",
        aspect="auto",
        template="plotly_white",
    )
    fig_heat.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        coloraxis_colorbar=dict(title=score_label),
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # 8. Top Keywords per Document (Bar Chart & Inspection)
    st.subheader("🏆 Top Keywords per Document")
    col_select, col_bars = st.columns([2, 3])

    with col_select:
        selected_doc_idx = st.selectbox(
            "Select Document to Inspect:",
            options=range(len(doc_lines)),
            format_func=lambda i: f"Doc {i+1}: {doc_lines[i][:45]}...",
        )
        chosen_doc_name = f"Doc {selected_doc_idx + 1}"
        full_doc_text = doc_lines[selected_doc_idx]

        st.markdown(
            f"""
            <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; 
                        border-radius: 8px; padding: 12px; margin-top: 10px;">
                <div style="font-weight: 600; font-size: 0.9rem; color: {COLOR_TEXT}; margin-bottom: 4px;">
                    Full Document Text:
                </div>
                <div style="color: {COLOR_MUTED}; font-size: 0.88rem; font-style: italic;">
                    "{full_doc_text}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Keyword Chips for top 3
        doc_scores = df_matrix.loc[chosen_doc_name]
        top_words = doc_scores[doc_scores > 0].sort_values(ascending=False).head(5)
        st.markdown("<br><b>Key Feature Tokens:</b>", unsafe_allow_html=True)
        chips_list = []
        for word, val in top_words.items():
            chips_list.append(chip(f"{word}: {val:.2f}" if decimals == 3 else f"{word}: {int(val)}", color=PASTEL_BLUE))
        st.markdown(f'<div class="chip-container">{" ".join(chips_list)}</div>', unsafe_allow_html=True)

    with col_bars:
        doc_scores = df_matrix.loc[chosen_doc_name]
        active_scores = doc_scores[doc_scores > 0].sort_values(ascending=True).tail(8)

        if not active_scores.empty:
            df_bar = pd.DataFrame({
                "Word": active_scores.index,
                "Score": active_scores.values,
            })
            fig_bar = px.bar(
                df_bar,
                x="Score",
                y="Word",
                orientation="h",
                text="Score",
                template="plotly_white",
                color_discrete_sequence=[COLOR_PRIMARY],
            )
            fig_bar.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                height=260,
                xaxis_title=score_label,
                yaxis_title=None,
                plot_bgcolor=COLOR_BG,
                paper_bgcolor=COLOR_BG,
            )
            fig_bar.update_traces(
                texttemplate="%{text:.2f}" if decimals == 3 else "%{text:d}",
                textposition="outside",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No words found for this document.")

    # 9. Educational Takeaway Box
    with st.expander("💡 Deep Dive: Mathematical Formulation of TF-IDF"):
        st.markdown(
            """
            * **Term Frequency ($TF$):**
              $$\\text{TF}(t, d) = \\frac{\\text{Count of word } t \\text{ in document } d}{\\text{Total words in document } d}$$
            * **Inverse Document Frequency ($IDF$):**
              $$\\text{IDF}(t) = \\ln\\left(\\frac{1 + N}{1 + \\text{DF}(t)}\\right) + 1$$
              *(where $N$ is total documents and $\\text{DF}(t)$ is number of documents containing $t$)*
            * **The Intuition:**
              If a word like `"processing"` appears only in Doc 1 and Doc 3, it has high discriminative power. 
              If every single document contains `"language"`, its IDF drops, preventing it from distinguishing between documents.
            """
        )
