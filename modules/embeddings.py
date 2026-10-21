"""
Module for Word Embeddings and 2D Vector Space Visualizer.
Implements Tab 6 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Semantic trajectory pairs for automatic arrow annotation in 2D PCA space
COMMON_ANALOGY_PAIRS = [
    ("man", "king", "👑 Royalty Shift (Man ➔ King)"),
    ("woman", "queen", "👑 Royalty Shift (Woman ➔ Queen)"),
    ("france", "paris", "🗼 Capital Shift (France ➔ Paris)"),
    ("japan", "tokyo", "🗼 Capital Shift (Japan ➔ Tokyo)"),
    ("germany", "berlin", "🗼 Capital Shift (Germany ➔ Berlin)"),
    ("italy", "rome", "🗼 Capital Shift (Italy ➔ Rome)"),
    ("big", "bigger", "📏 Comparative Shift (Big ➔ Bigger)"),
    ("cold", "colder", "📏 Comparative Shift (Cold ➔ Colder)"),
    ("apple", "banana", "🍎 Fruit Category"),
    ("car", "bus", "🚗 Vehicle Category"),
]


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

    # Dimensionality Selection: 2D vs 3D
    dim_col1, dim_col2 = st.columns([2, 3])
    with dim_col1:
        dim_choice = st.radio(
            "Select Visualization Dimension:",
            ["2D Projection (PCA)", "3D Interactive Orbit (PCA)"],
            horizontal=True,
            key="pca_dim_radio"
        )

    if "3D" in dim_choice:
        if len(valid_words) < 4:
            st.warning("⚠️ 3D PCA requires at least 4 valid words to construct a 3D coordinate space.")
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(np.array(vectors))
            df_plot = pd.DataFrame({"Word": valid_words, "PC 1": coords[:, 0], "PC 2": coords[:, 1]})
            fig = px.scatter(df_plot, x="PC 1", y="PC 2", text="Word", template="plotly_white", title="Fallback 2D PCA")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        else:
            pca_3d = PCA(n_components=3, random_state=42)
            coords_3d = pca_3d.fit_transform(np.array(vectors))
            var_3d = pca_3d.explained_variance_ratio_
            total_3d = sum(var_3d) * 100
            df_3d = pd.DataFrame({
                "Word": valid_words,
                "PC 1": coords_3d[:, 0],
                "PC 2": coords_3d[:, 1],
                "PC 3": coords_3d[:, 2],
            })
            fig_3d = px.scatter_3d(
                df_3d,
                x="PC 1",
                y="PC 2",
                z="PC 3",
                text="Word",
                template="plotly_white",
                title=f"3D Vector Space Orbit (Preserves {total_3d:.1f}% Variance)",
            )
            fig_3d.update_traces(
                textposition="top center",
                marker=dict(size=8, color=COLOR_PRIMARY, line=dict(width=1, color="#FFFFFF")),
            )
            fig_3d.update_layout(
                height=540,
                margin=dict(l=20, r=20, t=40, b=20),
                scene=dict(
                    xaxis_title="PC 1",
                    yaxis_title="PC 2",
                    zaxis_title="PC 3",
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption("🌐 **Interactive 3D Orbit:** Click and drag with mouse to freely rotate, spin, and zoom into semantic clusters.")
    else:
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

        # Trajectory arrow toggle
        show_arrows = st.checkbox(
            "🏹 Draw Semantic Analogy Shift Arrows (e.g. Man ➔ King, Woman ➔ Queen)",
            value=True,
            help="Renders directional vector displacement arrows between related word pairs in the 2D projection plane.",
            key="pca_show_arrows_chk"
        )

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

        # Add semantic trajectory arrows if pairs exist
        if show_arrows:
            for w_start, w_end, shift_label in COMMON_ANALOGY_PAIRS:
                if w_start in df_plot["Word"].values and w_end in df_plot["Word"].values:
                    p_start = df_plot[df_plot["Word"] == w_start].iloc[0]
                    p_end = df_plot[df_plot["Word"] == w_end].iloc[0]
                    arrow_col = "#4F46E5" if "Royalty" in shift_label else ("#059669" if "Capital" in shift_label else "#D97706")
                    fig.add_annotation(
                        x=p_end["PC 1"],
                        y=p_end["PC 2"],
                        ax=p_start["PC 1"],
                        ay=p_start["PC 2"],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        text=shift_label,
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.2,
                        arrowwidth=2.4,
                        arrowcolor=arrow_col,
                        font=dict(size=10, color=COLOR_TEXT, family="sans-serif"),
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor=COLOR_BORDER,
                        borderwidth=1,
                        borderpad=3,
                    )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Interactive:** Hover over dots, zoom, or pan to explore how conceptual clusters naturally emerge.")

    st.markdown("---")

    # 6. Section B: Pairwise Cosine Similarity Comparator
    st.subheader("📐 Word Pair Cosine Similarity")
    st.caption("Measure the cosine angle between any two 300-dimensional vector directions (range: -1.0 to +1.0):")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        word_a = st.text_input("First Word:", value="king", key="sim_word_a").strip().lower()
    with col_w2:
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

    st.markdown("---")

    # 7. Section C: Live Vector Arithmetic Playground (Analogy Solver)
    st.subheader("🧮 Semantic Vector Arithmetic: King − Man + Woman = ?")
    st.caption("Watch how dense embeddings capture linguistic analogies mathematically through vector addition and subtraction:")

    ANALOGY_PRESETS = {
        "👑 Classic Royalty: king − man + woman = queen": ("king", "man", "woman"),
        "🗼 Capitals: paris − france + japan = tokyo": ("paris", "france", "japan"),
        "📏 Comparative: bigger − big + cold = colder": ("bigger", "big", "cold"),
        "🏥 Occupations: doctor − man + woman = nurse/physician": ("doctor", "man", "woman"),
    }

    selected_analogy = st.selectbox(
        "Choose an Analogy Preset (or enter your own below):",
        options=list(ANALOGY_PRESETS.keys()),
        index=0,
    )
    def_w1, def_w2, def_w3 = ANALOGY_PRESETS[selected_analogy]

    c_a, c_sub, c_b, c_plus, c_c = st.columns([3, 1, 3, 1, 3])
    with c_a:
        wa = st.text_input("Word A (+):", value=def_w1, key="vec_math_a").strip().lower()
    with c_sub:
        st.markdown("<div style='text-align: center; font-size: 2rem; font-weight: 700; padding-top: 24px;'>−</div>", unsafe_allow_html=True)
    with c_b:
        wb = st.text_input("Word B (−):", value=def_w2, key="vec_math_b").strip().lower()
    with c_plus:
        st.markdown("<div style='text-align: center; font-size: 2rem; font-weight: 700; padding-top: 24px;'>+</div>", unsafe_allow_html=True)
    with c_c:
        wc = st.text_input("Word C (+):", value=def_w3, key="vec_math_c").strip().lower()

    if wa and wb and wc:
        t_a = nlp(wa)[0] if len(nlp(wa)) > 0 else None
        t_b = nlp(wb)[0] if len(nlp(wb)) > 0 else None
        t_c = nlp(wc)[0] if len(nlp(wc)) > 0 else None

        if not (t_a and t_a.has_vector and t_a.vector_norm > 0):
            st.error(f"'{wa}' is not in the vocabulary.")
        elif not (t_b and t_b.has_vector and t_b.vector_norm > 0):
            st.error(f"'{wb}' is not in the vocabulary.")
        elif not (t_c and t_c.has_vector and t_c.vector_norm > 0):
            st.error(f"'{wc}' is not in the vocabulary.")
        else:
            # Vector arithmetic: Target = A - B + C
            target_vec = t_a.vector - t_b.vector + t_c.vector
            target_norm = np.linalg.norm(target_vec)

            # Candidate pool of common English words for rapid vector search
            candidate_pool = [
                "queen", "king", "princess", "prince", "emperor", "monarch", "crown", "ruler", "palace",
                "man", "woman", "boy", "girl", "mother", "father", "son", "daughter", "sister", "brother",
                "tokyo", "japan", "paris", "france", "london", "england", "berlin", "germany", "rome", "italy",
                "colder", "cold", "warm", "warmer", "hot", "hotter", "bigger", "big", "small", "smaller", "fast", "faster",
                "nurse", "doctor", "physician", "surgeon", "hospital", "teacher", "engineer", "scientist", "actor", "actress",
                "apple", "banana", "fruit", "food", "car", "bus", "train", "vehicle", "city", "country", "president", "minister"
            ]
            # Exclude input words
            input_words = {wa, wb, wc}
            candidate_pool = [w for w in candidate_pool if w not in input_words]

            scores = []
            for cw in candidate_pool:
                c_tok = nlp(cw)[0]
                if c_tok and c_tok.has_vector and c_tok.vector_norm > 0:
                    sim = np.dot(target_vec, c_tok.vector) / (target_norm * c_tok.vector_norm)
                    scores.append((cw, float(sim)))

            scores.sort(key=lambda x: x[1], reverse=True)
            top_predictions = scores[:5]

            st.markdown("<br><b>🏆 Predicted Analogy Solutions (Ranked by Cosine Similarity):</b>", unsafe_allow_html=True)
            res_cols = st.columns(len(top_predictions))
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            for idx, (p_word, p_sim) in enumerate(top_predictions):
                with res_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; 
                                    border-radius: 10px; padding: 12px; text-align: center;">
                            <div style="font-size: 1.2rem;">{medals[idx]}</div>
                            <div style="font-size: 1.05rem; font-weight: 700; color: {COLOR_PRIMARY}; margin: 4px 0;">{p_word.title()}</div>
                            <div style="font-size: 0.85rem; color: {COLOR_MUTED};">{p_sim*100:.1f}% Match</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # --- 2D Analogy Vector Geometry: Parallelogram Law Visualizer ---
            if top_predictions:
                best_pred = top_predictions[0][0]
                t_pred = nlp(best_pred)[0]

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📐 2D Analogy Vector Geometry: The Parallelogram Law")
                st.markdown(
                    f"<span style='color: {COLOR_MUTED}; font-size: 0.9rem;'>"
                    f"Visualizing vector arithmetic in 2D PCA space: notice how the displacement vector "
                    f"<b>{wb.title()} → {wa.title()}</b> is nearly parallel and equal in length to "
                    f"<b>{wc.title()} → {best_pred.title()}</b>."
                    f"</span>",
                    unsafe_allow_html=True,
                )

                # Prepare 5 vectors: Base B, Anchor A, Base C, Predicted D, Ideal Target T
                geom_vecs = np.array([
                    t_b.vector,
                    t_a.vector,
                    t_c.vector,
                    t_pred.vector,
                    target_vec
                ])

                pca_geom = PCA(n_components=2, random_state=42)
                coords_geom = pca_geom.fit_transform(geom_vecs)

                # Unpack 2D points
                xb, yb = coords_geom[0]
                xa, ya = coords_geom[1]
                xc, yc = coords_geom[2]
                xd, yd = coords_geom[3]
                xt, yt = coords_geom[4]

                # Displacement vectors in 2D
                delta_1 = np.array([xa - xb, ya - yb])
                delta_2 = np.array([xd - xc, yd - yc])

                # Directional alignment (cosine similarity of 2D displacement vectors)
                norm_d1 = np.linalg.norm(delta_1)
                norm_d2 = np.linalg.norm(delta_2)
                alignment = (np.dot(delta_1, delta_2) / (norm_d1 * norm_d2)) if (norm_d1 > 0 and norm_d2 > 0) else 1.0
                alignment_pct = max(0.0, float(alignment) * 100)

                # Plotly Figure for Analogy Parallelogram
                fig_geom = go.Figure()

                # 1. Base Words Trace
                fig_geom.add_trace(go.Scatter(
                    x=[xb, xc],
                    y=[yb, yc],
                    mode="markers+text",
                    text=[f"<b>{wb.title()}</b> (Base 1)", f"<b>{wc.title()}</b> (Base 2)"],
                    textposition=["bottom center", "bottom center"],
                    marker=dict(size=15, color="#3B82F6", line=dict(width=2, color="#FFFFFF")),
                    name="Base Words (- / +)",
                ))

                # 2. Transformed & Predicted Words Trace
                fig_geom.add_trace(go.Scatter(
                    x=[xa, xd],
                    y=[ya, yd],
                    mode="markers+text",
                    text=[f"<b>{wa.title()}</b> (Anchor)", f"<b>★ {best_pred.title()}</b> (Top Prediction)"],
                    textposition=["top center", "top center"],
                    marker=dict(size=17, color="#10B981", line=dict(width=2, color="#FFFFFF")),
                    name="Target / Prediction",
                ))

                # 3. Ideal Theoretical Point Trace
                fig_geom.add_trace(go.Scatter(
                    x=[xt],
                    y=[yt],
                    mode="markers+text",
                    text=["<b>Ideal Arithmetic Target</b>"],
                    textposition="top right",
                    marker=dict(size=13, symbol="diamond", color="#6B7280", line=dict(width=1.5, color="#FFFFFF")),
                    name="Ideal Point (A - B + C)",
                ))

                # Arrow 1: Word B -> Word A
                fig_geom.add_annotation(
                    x=xa, y=ya,
                    ax=xb, ay=yb,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    text=f"Shift 1: ({wa} - {wb})",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.3,
                    arrowwidth=2.5,
                    arrowcolor="#2563EB",
                    font=dict(size=11, color=COLOR_TEXT, family="sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=COLOR_BORDER,
                    borderwidth=1,
                    borderpad=3,
                )

                # Arrow 2: Word C -> Word D (Actual Prediction)
                fig_geom.add_annotation(
                    x=xd, y=yd,
                    ax=xc, ay=yc,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    text=f"Shift 2: ({best_pred} - {wc})",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.3,
                    arrowwidth=2.5,
                    arrowcolor="#059669",
                    font=dict(size=11, color=COLOR_TEXT, family="sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=COLOR_BORDER,
                    borderwidth=1,
                    borderpad=3,
                )

                # Dashed Connector: Word C -> Ideal Target Point
                fig_geom.add_annotation(
                    x=xt, y=yt,
                    ax=xc, ay=yc,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    text="Theoretical Parallelogram Shift",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.0,
                    arrowwidth=1.8,
                    arrowcolor="#9CA3AF",
                    font=dict(size=10, color=COLOR_MUTED, family="sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor=COLOR_BORDER,
                    borderwidth=1,
                    borderpad=2,
                )

                # Dashed Connector from Predicted Word to Ideal Point (Error Gap)
                fig_geom.add_trace(go.Scatter(
                    x=[xd, xt],
                    y=[yd, yt],
                    mode="lines",
                    line=dict(color="#9CA3AF", width=1.5, dash="dot"),
                    name="Prediction-to-Ideal Gap",
                    hoverinfo="skip"
                ))

                fig_geom.update_layout(
                    title=f"Analogy Parallelogram: {wb.title()} → {wa.title()} vs. {wc.title()} → {best_pred.title()}",
                    template="plotly_white",
                    height=450,
                    margin=dict(l=30, r=30, t=50, b=30),
                    plot_bgcolor=COLOR_BG,
                    paper_bgcolor=COLOR_BG,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="Principal Component 1", showgrid=True, gridcolor=COLOR_BORDER, zeroline=False),
                    yaxis=dict(title="Principal Component 2", showgrid=True, gridcolor=COLOR_BORDER, zeroline=False),
                )

                st.plotly_chart(fig_geom, use_container_width=True)

                # Geometric Insights Cards
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1:
                    st.metric("Directional Parallelism", f"{alignment_pct:.1f}%", help="Cosine similarity between Shift 1 and Shift 2 in the 2D plane.")
                with g_col2:
                    gap_dist = np.linalg.norm(np.array([xd - xt, yd - yt]))
                    st.metric("Distance to Ideal Target", f"{gap_dist:.2f}", help="Euclidean distance between actual vocabulary word and ideal theoretical vector coordinates.")
                with g_col3:
                    angle_deg = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
                    st.metric("Parallelogram Angle", f"{angle_deg:.1f}°", help="Angular deviation between the two transformation arrows.")

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
