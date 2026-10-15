"""
Module for Stemming vs. Lemmatization: Porter, Snowball, and WordNet Lemmatizer.
Implements Tab 2 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import nltk
from utils.theme import (
    PASTEL_BLUE,
    PASTEL_PINK,
    PASTEL_GREEN,
    PASTEL_YELLOW,
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
