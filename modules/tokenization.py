"""
Module for Tokenization: Word, Sentence, and spaCy linguistic tokenization.
Implements Tab 1 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import nltk
from utils.theme import (
    PASTEL_COLORS,
    chip,
    COLOR_MUTED,
    COLOR_TEXT,
    COLOR_PRIMARY,
    COLOR_SECONDARY_BG,
    COLOR_BORDER
)
from utils.nlp_loader import ensure_nltk_data, load_spacy_model

# Default sample text demonstrating contractions, punctuation, and multiple sentences
DEFAULT_TEXT = (
    "Natural Language Processing (NLP) enables computers to understand, interpret, "
    "and manipulate human speech and text. Don't you think it's fascinating? "
    "In 2026, modern AI models can process millions of sentences in seconds!"
)


def render_tokenization_tab():
    """
    Renders the Tokenization visualizer interface.
    """
    # 1. Plain-English Concept Explanation
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1.05rem; color: {COLOR_TEXT}; margin-bottom: 4px;">
                ✂️ What is Tokenization?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.93rem; line-height: 1.5;">
                Tokenization is the foundational first step in any NLP pipeline. It breaks down raw, continuous 
                text into discrete, meaningful chunks called <b>tokens</b> (such as words, punctuation marks, or full sentences) 
                so algorithms can analyze and process them numerically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. User Input Area
    default_val = st.session_state.get("shared_input", DEFAULT_TEXT)
    user_text = st.text_area(
        label="Enter your sentence or paragraph to tokenize:",
        value=default_val,
        height=110,
        max_chars=1200,
        help="Type or paste any text. The app will tokenize it in real-time.",
        key="token_input_text_area",
    )

    # Edge Case: Empty or whitespace-only input
    if not user_text or not user_text.strip():
        st.info("👆 Please enter or paste some text above to see tokenization in action.")
        return

    # 3. Tokenizer Method Selection
    col_method, col_info = st.columns([2, 3])
    with col_method:
        method = st.radio(
            "Select Tokenizer Strategy:",
            [
                "NLTK Word Tokenizer (word_tokenize)",
                "NLTK Sentence Tokenizer (sent_tokenize)",
                "spaCy Linguistic Tokenizer",
            ],
            index=0,
            horizontal=False,
        )

    with col_info:
        if "Word Tokenizer" in method:
            st.caption(
                "**NLTK Word Tokenizer** splits text into individual words, contractions (`Don't` → `Do` + `n't`), "
                "and punctuation marks using Penn Treebank conventions."
            )
        elif "Sentence Tokenizer" in method:
            st.caption(
                "**NLTK Sentence Tokenizer** uses the Punkt model to split text into complete sentences while "
                "intelligently avoiding false splits on abbreviations (e.g., `Dr.`, `U.S.A.`)."
            )
        else:
            st.caption(
                "**spaCy Tokenizer** applies non-destructive rule-based linguistic tokenization that preserves whitespace "
                "and prepares tokens for part-of-speech tagging and parsing."
            )

    st.markdown("---")

    # 4. Tokenization Execution
    tokens = []
    token_details = []

    if "Word Tokenizer" in method:
        ensure_nltk_data()
        tokens = nltk.word_tokenize(user_text)
        for idx, tok in enumerate(tokens):
            token_details.append({
                "Index": idx + 1,
                "Token": tok,
                "Length": len(tok),
                "Category": "Punctuation" if not tok.isalnum() else ("Numeric" if tok.isnumeric() else "Word"),
                "Is Alphanumeric": tok.isalnum(),
            })

    elif "Sentence Tokenizer" in method:
        ensure_nltk_data()
        tokens = nltk.sent_tokenize(user_text)
        for idx, sent in enumerate(tokens):
            token_details.append({
                "Index": idx + 1,
                "Sentence Token": sent,
                "Word Count": len(sent.split()),
                "Character Length": len(sent),
            })

    else:  # spaCy
        nlp = load_spacy_model()
        doc = nlp(user_text)
        for idx, t in enumerate(doc):
            tokens.append(t.text)
            token_details.append({
                "Index": idx + 1,
                "Token": t.text,
                "Length": len(t.text),
                "Is Alpha": t.is_alpha,
                "Is Stopword": t.is_stop,
                "Is Punctuation": t.is_punct,
            })

    # 5. Metrics Summary Cards (with Lexical Diversity / TTR)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="Total Tokens", value=len(tokens))
    with m2:
        unique_tokens = len(set(tokens))
        st.metric(label="Unique Tokens", value=unique_tokens)
    with m3:
        avg_len = sum(len(t) for t in tokens) / max(len(tokens), 1)
        st.metric(label="Avg Token Length", value=f"{avg_len:.1f} chars")
    with m4:
        punct_count = sum(1 for t in tokens if not t.isalnum())
        st.metric(label="Punct / Special", value=punct_count)
    with m5:
        ttr = (unique_tokens / max(len(tokens), 1)) * 100
        st.metric(label="Lexical Diversity", value=f"{ttr:.1f}%", help="Type-Token Ratio (TTR): measures vocabulary richness.")

    # 6. Visual Output: Color-Coded Chips
    st.subheader("🎨 Live Token Stream")
    st.markdown(
        f"<span style='color: {COLOR_MUTED}; font-size: 0.88rem;'>"
        "Each token is rendered as an isolated pastel chip. Alternating colors show clear token boundaries:"
        "</span>",
        unsafe_allow_html=True,
    )

    chip_html_list = []
    for i, tok in enumerate(tokens):
        color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
        chip_html_list.append(chip(text=tok, color=color, subtitle=f"#{i+1}"))

    chips_container = f'<div class="chip-container">{" ".join(chip_html_list)}</div>'
    st.markdown(chips_container, unsafe_allow_html=True)

    # 7. Interactive N-Gram Exploration
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧩 N-Gram Phrase Inspector")
    st.caption("N-grams are contiguous sequences of N items from the text. They capture local context and collocations:")

    col_ngram_n, col_ngram_info = st.columns([2, 3])
    with col_ngram_n:
        ngram_choice = st.radio(
            "Select N-Gram Order:",
            ["Unigrams (N=1)", "Bigrams (N=2)", "Trigrams (N=3)"],
            horizontal=True,
            key="ngram_order_select"
        )
    
    n_order = 1 if "1" in ngram_choice else (2 if "2" in ngram_choice else 3)
    # Compute words-only tokens for meaningful n-grams
    clean_words = [t for t in tokens if t.isalnum()]
    
    if len(clean_words) >= n_order:
        from collections import Counter
        ngram_tuples = list(nltk.ngrams(clean_words, n_order))
        ngram_strings = [" ".join(ng) for ng in ngram_tuples]
        ngram_counts = Counter(ngram_strings).most_common(12)

        ngram_chips = []
        for idx, (ng_str, cnt) in enumerate(ngram_counts):
            color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
            ngram_chips.append(chip(ng_str, color=color, subtitle=f"{cnt}x"))
        
        st.markdown(f'<div class="chip-container">{" ".join(ngram_chips)}</div>', unsafe_allow_html=True)
    else:
        st.info(f"Please enter at least {n_order} words to generate {ngram_choice.split()[0]}.")

    # 8. Structured Table of Tokens
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Token Breakdown Table")
    df = pd.DataFrame(token_details)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 8. Educational Takeaway / Comparison Card
    with st.expander("💡 Deep Dive: How Different Tokenizers Treat Contractions & Symbols"):
        st.markdown(
            """
            * **Contractions:** Compare how words like `Don't` or `it's` are split.
              * Standard Python `.split()` gives: `["Don't", "you", "think", "it's"]` (keeps punctuation glued to words).
              * NLTK / spaCy tokenizers give: `["Do", "n't"]` and `["it", "'s"]`, separating the auxiliary verb from the negative particle `not` so downstream sentiment and grammar models understand both!
            * **Punctuation:** Real NLP tokenizers peel commas, periods, and quotation marks into separate tokens rather than treating `"seconds!"` as a distinct word from `"seconds"`.
            """
        )

    # 9. Optional Bonus: Marathi / Indic Script Tokenization (Task 6 Item 7)
    with st.expander("🇮🇳 Optional Bonus: Marathi / Hindi (Devanagari) Tokenization Demo"):
        st.markdown(
            "Explore how NLP handles non-Latin scripts like **Marathi** and **Hindi** where compound characters (जोडाक्षरे), "
            "vowel matras (मात्रा), and danda punctuation (`।`) require Unicode-aware segmentation:"
        )
        default_marathi = "नैसर्गिक भाषा प्रक्रिया (NLP) संगणकाला मानवी भाषा समजण्यास मदत करते. तुम्ही अभ्यास करत आहात का?"
        indic_text = st.text_input("Enter Marathi / Hindi text:", value=default_marathi, key="indic_token_input")
        if indic_text.strip():
            import re
            # Unicode-aware word and punctuation regex for Devanagari
            indic_tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+|[^\s\w]', indic_text, re.UNICODE)
            indic_chips = []
            for idx, itok in enumerate(indic_tokens):
                color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
                indic_chips.append(chip(itok, color=color, subtitle=f"#{idx+1}"))
            st.markdown(f'<div class="chip-container">{" ".join(indic_chips)}</div>', unsafe_allow_html=True)
            st.caption(f"Total Devanagari tokens extracted: **{len(indic_tokens)}**")
