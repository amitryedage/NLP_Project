"""
Module for Word Embeddings and 2D Vector Space Visualizer.
Implements Tab 6 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from utils.theme import (
    COLOR_BG,
    COLOR_SECONDARY_BG,
    COLOR_BORDER,
    COLOR_TEXT,
    COLOR_MUTED,
    COLOR_PRIMARY,
    chip,
    PASTEL_BLUE,
    PASTEL_GREEN,
    PASTEL_YELLOW,
    PASTEL_PINK,
    PASTEL_PURPLE,
    PASTEL_ORANGE,
)
from utils.nlp_loader import load_spacy_model

# Default benchmark word list as specified in PRD Task 5
DEFAULT_WORDS = "king, queen, man, woman, apple, banana, mango, car, bus, train"

PRESET_LISTS = {
    "👑 Royalty, Fruits & Vehicles (PRD Default)": "king, queen, man, woman, apple, banana, mango, car, bus, train",
    "🦁 Animals & Habitats": "lion, tiger, elephant, monkey, forest, jungle, ocean, river, fish, dolphin",
    "💻 Technology & Science": "computer, algorithm, software, code, physics, chemistry, biology, galaxy, star, planet",
    "☕ Daily Life & Refreshments": "coffee, tea, water, juice, bread, butter, cheese, breakfast, lunch, dinner",
}


def render_embeddings_tab():
    """
    Renders the Word Embeddings 2D PCA scatter plot and Cosine Similarity visualizer.
    """
    # 1. Plain-English Concept Explanation
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1.05rem; color: {COLOR_TEXT}; margin-bottom: 4px;">
                🌌 What are Word Embeddings?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.93rem; line-height: 1.5;">
                Word embeddings represent words as dense, high-dimensional numerical vectors (<b>300 coordinates per word</b> in spaCy's <code>en_core_web_md</code>). 
                Words used in similar grammatical and semantic contexts map to coordinates that sit close together in geometric space.
                <br>
                For example, the vector difference between <code>king</code> and <code>man</code> mirrors the difference between <code>queen</code> and <code>woman</code>!
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Section A: 2D Embeddings Scatter Plot
    st.subheader("🗺️ 2D Semantic Vector Space (PCA Projection)")

    # Preset dropdown
    selected_preset = st.selectbox(
        "Choose an example cluster preset (or customize your own below):",
        options=list(PRESET_LISTS.keys()),
        index=0,
    )

    user_words_input = st.text_input(
        label="Enter comma-separated words to project into vector space:",
        value=PRESET_LISTS[selected_preset],
        help="Enter at least 3 words separated by commas (e.g., king, queen, man, woman, car, bus).",
        key="embeddings_words_input",
    )

    # 3. Parse and Validate Words
    raw_words = [w.strip() for w in user_words_input.split(",") if w.strip()]

    if not raw_words:
        st.info("👆 Please enter at least 3 words separated by commas.")
        return

    # Load spaCy medium model
    nlp = load_spacy_model()

    # Extract 300D vectors and check Out-of-Vocabulary (OOV)
    valid_words = []
    vectors = []
    oov_words = []

    for word in raw_words:
        token = nlp(word)[0] if len(nlp(word)) > 0 else None
        if token and token.has_vector and token.vector_norm > 0:
            valid_words.append(token.text.lower())
            vectors.append(token.vector)
        else:
            oov_words.append(word)

    # Deduplicate while keeping vectors aligned
    unique_indices = []
    seen = set()
    for idx, w in enumerate(valid_words):
        if w not in seen:
            seen.add(w)
            unique_indices.append(idx)

    valid_words = [valid_words[i] for i in unique_indices]
    vectors = [vectors[i] for i in unique_indices]

    # OOV Warning Banner
    if oov_words:
        st.warning(
            f"⚠️ **Out-of-Vocabulary Words:** The following word(s) have no pre-trained vectors in `en_core_web_md`: "
            f"**{', '.join(oov_words)}**. They were excluded from the vector space."
        )

    # Minimum words requirement check (at least 3 valid words)
    if len(valid_words) < 3:
        st.warning(
            "⚠️ **Minimum 3 words required:** PCA needs at least 3 distinct valid words to construct a 2D projection space. "
            f"Currently only {len(valid_words)} valid word(s) found."
        )
        return

    # 4. Dimensionality Reduction via PCA (300D -> 2D)
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(np.array(vectors))
    var_explained = pca.explained_variance_ratio_
    total_var = (var_explained[0] + var_explained[1]) * 100

    # Build DataFrame for Plotly
    df_plot = pd.DataFrame({
        "Word": valid_words,
        "PC 1": coords_2d[:, 0],
        "PC 2": coords_2d[:, 1],
    })

    # 5. Interactive Plotly Scatter Plot
    fig = px.scatter(
        df_plot,
        x="PC 1",
        y="PC 2",
        text="Word",
        template="plotly_white",
        title=f"2D PCA Projection of 300D Vectors (Preserves {total_var:.1f}% Variance)",
    )
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=13, color=COLOR_TEXT, family="sans-serif"),
        marker=dict(
            size=14,
            color=COLOR_PRIMARY,
            line=dict(width=2, color="#FFFFFF"),
            opacity=0.9,
        ),
    )
    fig.update_layout(
        height=480,
        margin=dict(l=30, r=30, t=50, b=30),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        xaxis=dict(
            title=f"Principal Component 1 ({var_explained[0]*100:.1f}%)",
            showgrid=True,
            gridcolor=COLOR_BORDER,
            zeroline=True,
            zerolinecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            title=f"Principal Component 2 ({var_explained[1]*100:.1f}%)",
            showgrid=True,
            gridcolor=COLOR_BORDER,
            zeroline=True,
            zerolinecolor=COLOR_BORDER,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Interactive:** Hover over dots, zoom, or pan to explore how conceptual clusters naturally emerge.")

    st.markdown("---")

    # 6. Section B: Pairwise Cosine Similarity Comparator
    st.subheader("📐 Word Pair Cosine Similarity")
    st.caption("Measure the cosine angle between any two 300-dimensional vector directions (range: -1.0 to +1.0):")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        default_idx1 = 0 if len(valid_words) > 0 else 0
        word_a = st.text_input("First Word:", value="king", key="sim_word_a").strip().lower()
    with col_w2:
        default_idx2 = 1 if len(valid_words) > 1 else 0
        word_b = st.text_input("Second Word:", value="queen", key="sim_word_b").strip().lower()

    if word_a and word_b:
        token_a = nlp(word_a)[0] if len(nlp(word_a)) > 0 else None
        token_b = nlp(word_b)[0] if len(nlp(word_b)) > 0 else None

        a_has_vec = token_a and token_a.has_vector and token_a.vector_norm > 0
        b_has_vec = token_b and token_b.has_vector and token_b.vector_norm > 0

        if not a_has_vec:
            st.error(f"Word `'{word_a}'` is not in the vocabulary.")
        elif not b_has_vec:
            st.error(f"Word `'{word_b}'` is not in the vocabulary.")
        else:
            similarity = float(token_a.similarity(token_b))

            # Qualitative Interpretation
            if similarity >= 0.70:
                interp_color = PASTEL_GREEN
                interp_tag = "🔥 High Similarity"
                interp_desc = "Strongly related in meaning, domain, or common usage."
            elif similarity >= 0.40:
                interp_color = PASTEL_YELLOW
                interp_tag = "⚡ Moderate Similarity"
                interp_desc = "Related concept, same general field, or shared category."
            else:
                interp_color = PASTEL_PINK
                interp_tag = "❄️ Low Similarity"
                interp_desc = "Semantically distant or unrelated concepts."

            # Visual Result Card
            st.markdown(
                f"""
                <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; 
                            border-radius: 12px; padding: 20px; margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <span style="font-size: 1.25rem; font-weight: 600; color: {COLOR_TEXT};">
                                <b>{word_a.title()}</b> ↔ <b>{word_b.title()}</b>
                            </span>
                        </div>
                        <div>
                            <span style="background-color: {interp_color}; color: {COLOR_TEXT}; padding: 6px 14px; 
                                         border-radius: 20px; font-weight: 600; font-size: 0.95rem;">
                                {interp_tag}
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {COLOR_PRIMARY}; margin: 12px 0 6px 0;">
                        {similarity:.4f}
                    </div>
                    <div style="color: {COLOR_MUTED}; font-size: 0.93rem;">
                        {interp_desc} (Cosine of the angle between both 300-dimensional vectors)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 7. Quick Benchmark Comparison Buttons
    st.markdown("<br><b>Try Classic Semantic Pairs:</b>", unsafe_allow_html=True)
    bench_cols = st.columns(4)
    with bench_cols[0]:
        st.caption("👑 `king` ↔ `queen` (~0.72)")
    with bench_cols[1]:
        st.caption("🍎 `apple` ↔ `banana` (~0.68)")
    with bench_cols[2]:
        st.caption("🚗 `car` ↔ `train` (~0.58)")
    with bench_cols[3]:
        st.caption("👑 `king` ↔ `apple` (~0.12)")

    # 8. Educational Deep Dive Box
    with st.expander("💡 Deep Dive: How Word Embeddings & Cosine Similarity Work"):
        st.markdown(
            """
            * **The Distributional Hypothesis:** *"You shall know a word by the company it keeps"* (J.R. Firth, 1957).
              Because words like `apple`, `banana`, and `mango` appear near verbs like *eat*, *peel*, and *fresh*, their 300-dimensional vectors converge into a shared fruit cluster.
            * **Cosine Similarity Formula:**
              $$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|} = \\frac{\\sum_{i=1}^{300} u_i v_i}{\\sqrt{\\sum_{i=1}^{300} u_i^2} \\sqrt{\\sum_{i=1}^{300} v_i^2}}$$
              * Unlike Euclidean distance, cosine similarity measures **directional alignment** rather than vector magnitude, making it invariant to word frequency.
            * **Why 300 Dimensions?**
              Natural language requires hundreds of latent features (e.g., gender, animacy, grammatical role, abstractness) to express rich semantic relationships accurately.
            """
        )
