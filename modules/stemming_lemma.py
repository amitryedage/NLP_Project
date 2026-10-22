"""
Module for Stemming vs. Lemmatization: Porter, Snowball, and WordNet Lemmatizer.
Implements Tab 2 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import nltk
from utils.theme import (
    COLOR_BG,
    PASTEL_BLUE,
    PASTEL_PINK,
    PASTEL_GREEN,
    PASTEL_YELLOW,
    PASTEL_PURPLE,
    PASTEL_ORANGE,
    chip,
    COLOR_MUTED,
    COLOR_TEXT,
    COLOR_SECONDARY_BG,
    COLOR_BORDER,
    COLOR_PRIMARY,
)
from utils.nlp_loader import get_stemmers_and_lemmatizers, ensure_nltk_data

# Default sample text highlighting common inflectional and irregular morphological cases
DEFAULT_SENTENCE = (
    "The experienced researchers were studying flying bats, green leaves, "
    "and various connections while running happily on their feet."
)

DEFAULT_WORDS = "running, studies, leaves, flies, easily, fairly, connection, computing, better, mice, feet"


def render_stemming_lemma_tab():
    """
    Renders the Stemming vs. Lemmatization comparative visualizer.
    """
    # 1. Plain-English Concept Explanation
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1.05rem; color: {COLOR_TEXT}; margin-bottom: 4px;">
                🌱 Stemming vs. Lemmatization: What's the Difference?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.93rem; line-height: 1.5;">
                Both techniques reduce words to their root forms, but use fundamentally different approaches:
                <br>
                • <b>Stemming:</b> Fast, heuristic algorithm that chops off prefixes/suffixes (e.g., <i>-ing, -ed, -es</i>). 
                  Often produces crude, non-dictionary base stems like <code>studi</code> or <code>easili</code>.
                <br>
                • <b>Lemmatization:</b> Lexical analysis using vocabulary and morphological grammar rules to return 
                  the actual valid dictionary base word (called a <b>lemma</b>), such as <i>study</i> or <i>leaf</i>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Input Mode Toggle & Text Area
    col_mode, col_preset = st.columns([2, 3])
    with col_mode:
        input_type = st.radio(
            "Select Input Type:",
            ["Full Sentence", "Comma-Separated Words"],
            horizontal=True,
        )

    if input_type == "Full Sentence":
        default_val = st.session_state.get("shared_input", DEFAULT_SENTENCE)
        user_input = st.text_area(
            "Enter sentence to compare roots:",
            value=default_val,
            height=90,
            max_chars=1000,
            help="Every word in the sentence will be extracted and tested through each stemmer and lemmatizer.",
            key="stem_input_sentence",
        )
    else:
        user_input = st.text_area(
            "Enter comma-separated words:",
            value=DEFAULT_WORDS,
            height=90,
            max_chars=1000,
            help="Enter words separated by commas (e.g. running, studies, leaves).",
        )

    # Edge Case: Empty input
    if not user_input or not user_input.strip():
        st.info("👆 Please enter words or a sentence above to see the comparison.")
        return

    # 3. Extract Words
    ensure_nltk_data()
    stemmer_pack = get_stemmers_and_lemmatizers()
    porter = stemmer_pack["porter"]
    snowball = stemmer_pack["snowball"]
    lemmatizer = stemmer_pack["lemmatizer"]

    if input_type == "Full Sentence":
        # Tokenize sentence and filter to alphabetic words
        raw_tokens = nltk.word_tokenize(user_input)
        words = [w for w in raw_tokens if any(c.isalpha() for c in w)]
    else:
        # Split on commas
        words = [w.strip() for w in user_input.split(",") if w.strip()]

    if not words:
        st.warning("No valid words found to process. Please enter alphabetic words.")
        return

    # Deduplicate while preserving order for cleaner comparison
    unique_words = list(dict.fromkeys(words))

    # 4. Process Words through Algorithms
    comparison_data = []
    differing_cases = []

    for word in unique_words:
        w_lower = word.lower()
        p_stem = porter.stem(w_lower)
        s_stem = snowball.stem(w_lower)
        
        # WordNet noun & verb lemmatization
        lemma_n = lemmatizer.lemmatize(w_lower, pos="n")
        lemma_v = lemmatizer.lemmatize(w_lower, pos="v")
        
        # Determine best representative lemma
        if lemma_v != w_lower and (lemma_n == w_lower or len(lemma_v) < len(lemma_n)):
            best_lemma = lemma_v
            lemma_pos = "Verb"
        elif lemma_n != w_lower:
            best_lemma = lemma_n
            lemma_pos = "Noun"
        else:
            best_lemma = lemma_n
            lemma_pos = "Default"

        # Check if stem differs from the true lemma
        has_difference = (p_stem != best_lemma) or (s_stem != best_lemma)

        if has_difference:
            differing_cases.append({
                "word": word,
                "stem": p_stem,
                "lemma": best_lemma,
            })

        comparison_data.append({
            "Word": word,
            "Porter Stem": p_stem,
            "Snowball Stem": s_stem,
            "WordNet Lemma": best_lemma,
            "Lemma POS": lemma_pos,
            "Stem ≠ Lemma": "⚠️ Differ" if has_difference else "✅ Match",
        })

    # 5. Metrics Summary
    diff_count = len(differing_cases)
    total_count = len(unique_words)
    diff_pct = (diff_count / total_count * 100) if total_count > 0 else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Unique Words Tested", total_count)
    with m2:
        st.metric("Discrepancies (Stem ≠ Lemma)", f"{diff_count} words")
    with m3:
        st.metric("Discrepancy Rate", f"{diff_pct:.0f}%")

    st.markdown("---")

    # 6. Highlighted Discrepancy Chips
    st.subheader("🔍 Key Morphological Discrepancies")
    if differing_cases:
        st.markdown(
            f"<span style='color: {COLOR_MUTED}; font-size: 0.88rem;'>"
            "Notice how stemmers chop characters into crude stems, while the lemmatizer preserves valid vocabulary:"
            "</span>",
            unsafe_allow_html=True,
        )
        highlight_chips = []
        for case in differing_cases:
            chip_label = f"{case['word']} → stem: {case['stem']} | lemma: {case['lemma']}"
            highlight_chips.append(chip(chip_label, color=PASTEL_YELLOW))
        
        st.markdown(
            f'<div class="chip-container">{" ".join(highlight_chips)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("All tested words produced matching stems and lemmas!")

    # 7. Side-by-Side Comparative Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Side-by-Side Comparison Table")
    df_compare = pd.DataFrame(comparison_data)

    # Style table to highlight differences
    def highlight_diff(val):
        if val == "⚠️ Differ":
            return f"background-color: {PASTEL_PINK}; color: {COLOR_TEXT}; font-weight: 600;"
        return f"background-color: {PASTEL_GREEN}; color: {COLOR_TEXT};"

    # Support pandas >= 2.1 Styler.map and fallback for older versions
    styler = df_compare.style
    if hasattr(styler, "map"):
        styled_df = styler.map(highlight_diff, subset=["Stem ≠ Lemma"])
    else:
        styled_df = styler.applymap(highlight_diff, subset=["Stem ≠ Lemma"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Export Comparison Table Button
    st.download_button(
        label="📥 Download Comparison Table (CSV)",
        data=df_compare.to_csv(index=False).encode("utf-8"),
        file_name="stem_vs_lemma_comparison.csv",
        mime="text/csv",
        key="btn_download_stem_compare"
    )

    # 7b. Minimum Edit Distance (Levenshtein Distance) Matrix Visualizer
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📏 Minimum Edit Distance (Levenshtein Matrix)")
    st.caption("Measures morphological dissimilarity by calculating the minimum cost alignment (insertions, deletions, substitutions):")

    # Presets for quick comparison
    col_ep1, col_ep2, col_ep3, col_ep4 = st.columns(4)
    with col_ep1:
        if st.button("Preset: 'studies' ➔ 'study'", use_container_width=True, key="btn_ed_p1"):
            st.session_state["ed_src"] = "studies"
            st.session_state["ed_tgt"] = "study"
    with col_ep2:
        if st.button("Preset: 'studies' ➔ 'studi'", use_container_width=True, key="btn_ed_p2"):
            st.session_state["ed_src"] = "studies"
            st.session_state["ed_tgt"] = "studi"
    with col_ep3:
        if st.button("Preset: 'running' ➔ 'run'", use_container_width=True, key="btn_ed_p3"):
            st.session_state["ed_src"] = "running"
            st.session_state["ed_tgt"] = "run"
    with col_ep4:
        if st.button("Preset: 'intention' ➔ 'execution'", use_container_width=True, key="btn_ed_p4"):
            st.session_state["ed_src"] = "intention"
            st.session_state["ed_tgt"] = "execution"

    default_src = st.session_state.get("ed_src", "studies")
    default_tgt = st.session_state.get("ed_tgt", "study")

    col_w_src, col_w_tgt = st.columns(2)
    with col_w_src:
        word_src = st.text_input("Source Word (e.g. Inflected Form):", value=default_src, key="input_ed_src").strip().lower()
    with col_w_tgt:
        word_tgt = st.text_input("Target Word (e.g. Stem or Lemma):", value=default_tgt, key="input_ed_tgt").strip().lower()

    if word_src and word_tgt:
        m, n = len(word_src), len(word_tgt)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                sub_cost = 0 if word_src[i - 1] == word_tgt[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,        # deletion
                    dp[i][j - 1] + 1,        # insertion
                    dp[i - 1][j - 1] + sub_cost  # match / substitution
                )

        edit_dist = int(dp[m][n])
        max_len = max(m, n, 1)
        sim_ratio = (1.0 - (edit_dist / max_len)) * 100

        # Metric summary row
        ed_m1, ed_m2, ed_m3 = st.columns(3)
        with ed_m1:
            st.metric("Minimum Edit Distance", f"{edit_dist} operations")
        with ed_m2:
            st.metric("Morphological Similarity", f"{sim_ratio:.1f}%")
        with ed_m3:
            st.metric("Length Differential", f"|{m} - {n}| = {abs(m - n)} chars")

        # DP Grid Heatmap
        row_labels = ["#"] + list(word_src)
        col_labels = ["#"] + list(word_tgt)

        df_dp = pd.DataFrame(dp, index=row_labels, columns=col_labels)

        fig_dp = px.imshow(
            df_dp,
            labels=dict(x=f"Target: '{word_tgt}'", y=f"Source: '{word_src}'", color="Edit Cost"),
            x=col_labels,
            y=row_labels,
            color_continuous_scale="Purples",
            text_auto=True,
            template="plotly_white",
        )
        fig_dp.update_layout(
            title=f"Dynamic Programming Grid: '{word_src}' ➔ '{word_tgt}' (Cost = {edit_dist})",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            plot_bgcolor=COLOR_BG,
            paper_bgcolor=COLOR_BG,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_dp, use_container_width=True)

    # 7c. Morphological Morpheme Anatomy & Affix Decomposition Inspector
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧬 Morphological Morpheme Anatomy (Inflectional vs. Derivational)")
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: {COLOR_TEXT}; margin-bottom: 3px;">
                Morpheme Anatomy: How Words are Built
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.88rem; line-height: 1.4;">
                Words are constructed from <b>morphemes</b> (the smallest units of linguistic meaning). 
                Morphology distinguishes between:
                <br>
                • <b>Inflectional Affixes:</b> Modifies grammatical tense, number, or aspect without changing the word class (e.g. <i>play</i> [V] ➔ <i>playing</i> [V]).
                <br>
                • <b>Derivational Affixes:</b> Creates entirely new words, frequently changing the grammatical class (e.g. <i>happy</i> [Adj] ➔ <i>unhappiness</i> [N]).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Presets for Morphological Decomposition
    c_mp1, c_mp2, c_mp3, c_mp4 = st.columns(4)
    with c_mp1:
        if st.button("Word: 'unhappiness'", use_container_width=True, key="btn_morph_unhappy"):
            st.session_state["morph_word_input"] = "unhappiness"
    with c_mp2:
        if st.button("Word: 'transformations'", use_container_width=True, key="btn_morph_transform"):
            st.session_state["morph_word_input"] = "transformations"
    with c_mp3:
        if st.button("Word: 'rebuilding'", use_container_width=True, key="btn_morph_rebuild"):
            st.session_state["morph_word_input"] = "rebuilding"
    with c_mp4:
        if st.button("Word: 'connections'", use_container_width=True, key="btn_morph_connect"):
            st.session_state["morph_word_input"] = "connections"

    default_morph = st.session_state.get("morph_word_input", "unhappiness")
    morph_word = st.text_input("Enter a complex morphological word to decompose:", value=default_morph, key="morph_text_input").strip().lower()

    if morph_word:
        # Morphological Rule Database for common prefixes, derivational and inflectional suffixes
        PREFIXES = {
            "un": ("Derivational Prefix", "Reversal / Negation (e.g. un-happy)"),
            "re": ("Derivational Prefix", "Repetition / Again (e.g. re-build)"),
            "dis": ("Derivational Prefix", "Negation / Separation (e.g. dis-connect)"),
            "in": ("Derivational Prefix", "Negation / Inward (e.g. in-accurate)"),
            "im": ("Derivational Prefix", "Negation (e.g. im-possible)"),
            "trans": ("Derivational Prefix", "Across / Through (e.g. trans-form)"),
            "pre": ("Derivational Prefix", "Beforehand (e.g. pre-view)"),
            "anti": ("Derivational Prefix", "Opposing / Against (e.g. anti-virus)"),
            "mis": ("Derivational Prefix", "Wrongly / Badly (e.g. mis-understand)"),
        }

        INFLECTIONAL_SUFFIXES = {
            "ing": ("Inflectional Suffix", "Present Participle / Continuous Aspect"),
            "ed": ("Inflectional Suffix", "Past Tense / Past Participle"),
            "es": ("Inflectional Suffix", "Plural Noun / 3rd Person Singular Verb"),
            "s": ("Inflectional Suffix", "Plural Noun / 3rd Person Singular Verb"),
            "er": ("Inflectional Suffix", "Comparative Degree (Adj) or Agentive"),
            "est": ("Inflectional Suffix", "Superlative Degree (Adj)"),
        }

        DERIVATIONAL_SUFFIXES = {
            "ation": ("Derivational Suffix", "Verb ➔ Noun (Action / Process)"),
            "tion": ("Derivational Suffix", "Verb ➔ Noun (State / Action)"),
            "ness": ("Derivational Suffix", "Adjective ➔ Noun (Quality / State)"),
            "ment": ("Derivational Suffix", "Verb ➔ Noun (Action / Result)"),
            "able": ("Derivational Suffix", "Verb ➔ Adjective (Capability)"),
            "ible": ("Derivational Suffix", "Verb ➔ Adjective (Capability)"),
            "ity": ("Derivational Suffix", "Adjective ➔ Noun (State / Property)"),
            "ful": ("Derivational Suffix", "Noun ➔ Adjective (Full of)"),
            "less": ("Derivational Suffix", "Noun ➔ Adjective (Without)"),
            "ize": ("Derivational Suffix", "Noun/Adj ➔ Verb (Cause to be)"),
            "ise": ("Derivational Suffix", "Noun/Adj ➔ Verb (Cause to be)"),
            "ly": ("Derivational Suffix", "Adjective ➔ Adverb (Manner)"),
            "al": ("Derivational Suffix", "Noun ➔ Adjective (Relating to)"),
        }

        # Step 1: Detect Prefix
        detected_prefix = None
        rem_word = morph_word
        for p, (p_type, p_desc) in sorted(PREFIXES.items(), key=lambda x: len(x[0]), reverse=True):
            if rem_word.startswith(p) and len(rem_word) > len(p) + 2:
                detected_prefix = (p, p_type, p_desc)
                rem_word = rem_word[len(p):]
                break

        # Step 2: Detect Inflectional Suffix
        detected_infl = None
        for inf, (inf_type, inf_desc) in sorted(INFLECTIONAL_SUFFIXES.items(), key=lambda x: len(x[0]), reverse=True):
            if rem_word.endswith(inf) and len(rem_word) > len(inf) + 2:
                detected_infl = (inf, inf_type, inf_desc)
                rem_word = rem_word[:-len(inf)]
                break

        # Step 3: Detect Derivational Suffix
        detected_deriv = None
        for der, (der_type, der_desc) in sorted(DERIVATIONAL_SUFFIXES.items(), key=lambda x: len(x[0]), reverse=True):
            if rem_word.endswith(der) and len(rem_word) > len(der) + 2:
                detected_deriv = (der, der_type, der_desc)
                rem_word = rem_word[:-len(der)]
                break

        # Base Root
        base_root = rem_word
        # Lemmatize base root
        root_lemma = lemmatizer.lemmatize(base_root)
        if root_lemma.endswith('i'):
            root_lemma = root_lemma[:-1] + 'y'

        # Build Chips & Records
        morpheme_chips = []
        morph_records = []

        if detected_prefix:
            p_text, p_type, p_desc = detected_prefix
            morpheme_chips.append(chip(f"{p_text}-", color=PASTEL_PINK, subtitle=p_type))
            morph_records.append({"Morpheme": f"{p_text}-", "Role": "Prefix", "Morphological Category": p_type, "Linguistic Function": p_desc})

        morpheme_chips.append(chip(root_lemma, color=PASTEL_GREEN, subtitle="Root Lemma"))
        morph_records.append({"Morpheme": root_lemma, "Role": "Root / Base", "Morphological Category": "Free Lexical Morpheme", "Linguistic Function": "Core semantic concept foundation"})

        if detected_deriv:
            d_text, d_type, d_desc = detected_deriv
            morpheme_chips.append(chip(f"-{d_text}", color=PASTEL_PURPLE, subtitle=d_type))
            morph_records.append({"Morpheme": f"-{d_text}", "Role": "Derivational Suffix", "Morphological Category": d_type, "Linguistic Function": d_desc})

        if detected_infl:
            i_text, i_type, i_desc = detected_infl
            morpheme_chips.append(chip(f"-{i_text}", color=PASTEL_YELLOW, subtitle=i_type))
            morph_records.append({"Morpheme": f"-{i_text}", "Role": "Inflectional Suffix", "Morphological Category": i_type, "Linguistic Function": i_desc})

        st.markdown(f"**Morpheme Anatomy of '{morph_word}':**")
        st.markdown(f'<div class="chip-container">{" ".join(morpheme_chips)}</div>', unsafe_allow_html=True)

        df_morph = pd.DataFrame(morph_records)
        st.dataframe(df_morph, use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 Download Morpheme Breakdown (CSV)",
            data=df_morph.to_csv(index=False).encode("utf-8"),
            file_name=f"morpheme_anatomy_{morph_word}.csv",
            mime="text/csv",
            key="btn_download_morph_table"
        )

    # 8. Educational Takeaway Box
    with st.expander("💡 Deep Dive: Porter vs. Snowball vs. WordNet Rules"):
        st.markdown(
            """
            * **Why `studies` becomes `studi` in stemming:**
              * Porter and Snowball use algorithmic regex rules like `(*v*) ies -> i`. They don't check whether `studi` is a real English dictionary word.
            * **Why Lemmatization requires Part-of-Speech (POS):**
              * The word `leaves` could be the plural of `leaf` (Noun), or the 3rd person singular of `leave` (Verb). WordNet uses vocabulary databases to output real dictionary forms.
            * **Irregular plurals (`mice` → `mouse`, `feet` → `foot`):**
              * Stemmers completely fail on irregulars because no standard suffix is stripped (`feet` stays `feet`). Lemmatizers easily map irregulars to their proper singular base!
            """
        )
