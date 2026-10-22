"""
Module for Tokenization: Word, Sentence, and spaCy linguistic tokenization.
Implements Tab 1 of NLP Visual Lab adhering to PRD specifications.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import nltk
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from utils.theme import (
    COLOR_BG,
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

    # Stopword Filtering Controls
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        filter_stopwords = st.checkbox("🚫 Highlight English Stopwords (e.g., 'the', 'is', 'in')", value=False, key="tok_filter_stop")
    with col_f2:
        hide_stopwords = st.checkbox("Hide Stopwords (Keep Content Words Only)", value=False, key="tok_hide_stop") if filter_stopwords else False

    st.markdown("---")

    # 4. Tokenization Execution
    tokens = []
    token_details = []

    if "Word Tokenizer" in method:
        ensure_nltk_data()
        tokens = nltk.word_tokenize(user_text)
        for idx, tok in enumerate(tokens):
            is_stop = tok.lower() in SPACY_STOP_WORDS
            token_details.append({
                "Index": idx + 1,
                "Token": tok,
                "Length": len(tok),
                "Category": "Punctuation" if not tok.isalnum() else ("Numeric" if tok.isnumeric() else "Word"),
                "Is Alphanumeric": tok.isalnum(),
                "Is Stopword": is_stop,
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
        for idx, tok in enumerate(doc):
            tokens.append(tok.text)
            token_details.append({
                "Index": idx + 1,
                "Token": tok.text,
                "Length": len(tok.text),
                "Is Alpha": tok.is_alpha,
                "Is Stopword": tok.is_stop,
                "Is Punctuation": tok.is_punct,
            })

    # 5. Metrics Summary Cards (with Lexical Diversity / TTR & Stopword stats)
    stopword_count = sum(1 for d in token_details if d.get("Is Stopword", False))
    content_count = len(tokens) - stopword_count

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="Total Tokens", value=len(tokens))
    with m2:
        unique_tokens = len(set(tokens))
        st.metric(label="Unique Tokens", value=unique_tokens)
    with m3:
        if filter_stopwords:
            st.metric(label="Content Tokens", value=content_count)
        else:
            avg_len = sum(len(t) for t in tokens) / max(len(tokens), 1)
            st.metric(label="Avg Token Length", value=f"{avg_len:.1f} chars")
    with m4:
        if filter_stopwords:
            stop_pct = (stopword_count / max(len(tokens), 1)) * 100
            st.metric(label="Stopwords Filtered", value=f"{stopword_count} ({stop_pct:.0f}%)")
        else:
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
        is_stop = tok.lower() in SPACY_STOP_WORDS
        if hide_stopwords and is_stop:
            continue
        if filter_stopwords and is_stop:
            chip_html_list.append(chip(text=tok, color="#F3F4F6", subtitle="STOP"))
        else:
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

    # 7b. Zipf's Law Linguistic Distribution
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Zipf's Law Linguistic Distribution")
    st.caption("Empirical word frequency vs. rank ($f(r) \\propto 1/r$). Zipf's law predicts a linear drop on log-log coordinates:")

    zipf_words = [t.lower() for t in tokens if t.isalnum()]
    if len(zipf_words) >= 4:
        from collections import Counter
        word_counts = Counter(zipf_words).most_common()
        ranks = list(range(1, len(word_counts) + 1))
        words_list = [w for w, _ in word_counts]
        freqs = [cnt for _, cnt in word_counts]

        # Theoretical Zipf curve: f(1) / r
        f1 = freqs[0]
        theoretical_zipf = [f1 / r for r in ranks]

        df_zipf = pd.DataFrame({
            "Rank": ranks,
            "Word": words_list,
            "Empirical Frequency": freqs,
            "Theoretical Zipf (f1/r)": theoretical_zipf,
        })

        col_scale, _ = st.columns([2, 3])
        with col_scale:
            use_log = st.checkbox("Log-Log Scale Coordinates (Linearizes Zipf's curve)", value=True, key="zipf_log_toggle")

        fig_zipf = px.scatter(
            df_zipf,
            x="Rank",
            y="Empirical Frequency",
            text="Word",
            template="plotly_white",
            title=f"Zipf's Law Distribution for Top {len(df_zipf)} Vocabulary Words",
            log_x=use_log,
            log_y=use_log,
            color_discrete_sequence=[COLOR_PRIMARY]
        )
        fig_zipf.add_scatter(
            x=df_zipf["Rank"],
            y=df_zipf["Theoretical Zipf (f1/r)"],
            mode="lines",
            name="Ideal Zipf Curve (1/r)",
            line=dict(color="#10B981", dash="dash", width=2)
        )
        fig_zipf.update_traces(
            textposition="top center",
            textfont=dict(size=11, color=COLOR_TEXT)
        )
        fig_zipf.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor=COLOR_BG,
            paper_bgcolor=COLOR_BG,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_zipf, use_container_width=True)
    else:
        st.info("Enter at least 4 words to generate Zipf's Law rank distribution.")

    # 7b. Hierarchical Linguistic Word Treemap
    with st.expander("🗺️ Hierarchical Linguistic Word Treemap (Interactive Word Cloud Alternative)", expanded=False):
        st.markdown(
            f"<span style='color: {COLOR_MUTED}; font-size: 0.9rem;'>"
            f"A modern, interactive hierarchical alternative to cluttered static word clouds. "
            f"Tokens are structured by linguistic classification (Content vs. Stopwords) and token length, "
            f"with rectangle dimensions proportional to empirical frequency:"
            f"</span>",
            unsafe_allow_html=True,
        )
        tree_words = [t.lower() for t in tokens if any(c.isalnum() for c in t)]
        from collections import Counter
        word_freq = Counter(tree_words)
        if len(word_freq) >= 3:
            treemap_records = []
            for w, count in word_freq.items():
                is_stop = w.lower() in SPACY_STOP_WORDS
                cat = "Stopwords (Grammatical)" if is_stop else "Content Words (Lexical)"
                length_cat = "Short (≤3)" if len(w) <= 3 else ("Medium (4-7)" if len(w) <= 7 else "Long (≥8)")
                treemap_records.append({
                    "Word": w,
                    "Category": cat,
                    "Length Group": length_cat,
                    "Frequency": count,
                    "Token Length": len(w)
                })
            df_treemap = pd.DataFrame(treemap_records)
            fig_tree = px.treemap(
                df_treemap,
                path=["Category", "Length Group", "Word"],
                values="Frequency",
                color="Token Length",
                color_continuous_scale="Purples",
                title="Hierarchical Linguistic Word Treemap",
            )
            fig_tree.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=35, b=10),
                plot_bgcolor=COLOR_BG,
                paper_bgcolor=COLOR_BG
            )
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("Enter more words to generate the hierarchical treemap.")

    # 8. Interactive N-Gram Language Model & Next-Word Markov Predictor
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔮 N-Gram Language Model & Next-Word Predictor (Markov LM)")
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SECONDARY_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: {COLOR_TEXT}; margin-bottom: 3px;">
                How do Statistical Language Models Predict the Next Word?
            </div>
            <div style="color: {COLOR_MUTED}; font-size: 0.88rem; line-height: 1.4;">
                Before Neural Transformers (GPT), NLP relied on <b>N-gram Markov Language Models</b>. 
                Using the <i>Markov Assumption</i>, the probability of the next word <i>w<sub>t</sub></i> depends only on the preceding <i>N-1</i> words.
                Explore empirical transition probabilities and test next-word autocompletion directly on your input text:
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"P(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}")

    # Clean words for language modeling (lowercase alphanumeric words)
    import re
    lm_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\'-]+\b', user_text)]
    vocab_set = sorted(list(set(lm_words)))

    if len(lm_words) >= 4:
        c_lm1, c_lm2, c_lm3 = st.columns([1, 1, 1])
        with c_lm1:
            ngram_order = st.radio("Model Order:", ["Bigram (N = 2)", "Trigram (N = 3)"], index=0, horizontal=True)
        with c_lm2:
            smoothing = st.radio("Probability Estimation:", ["MLE (Maximum Likelihood)", "Laplace (Add-1 Smoothing)"], index=0)
        with c_lm3:
            st.metric("Model Vocabulary Size (|V|)", f"{len(vocab_set)} unique words")

        # Build N-Gram frequencies
        from collections import defaultdict, Counter
        if "Bigram" in ngram_order:
            bigram_counts = defaultdict(Counter)
            unigram_counts = Counter(lm_words)
            for w1, w2 in zip(lm_words[:-1], lm_words[1:]):
                bigram_counts[w1][w2] += 1

            # Word prompt selector
            available_contexts = [w for w in vocab_set if w in bigram_counts and len(bigram_counts[w]) > 0]
            if not available_contexts:
                available_contexts = vocab_set

            sel_col1, sel_col2 = st.columns([1, 2])
            with sel_col1:
                selected_prompt = st.selectbox("Select Prompt Word (History):", available_contexts, index=0)
            
            # Compute probabilities for selected prompt
            candidates = bigram_counts[selected_prompt]
            c_prev = unigram_counts[selected_prompt]
            v_size = len(vocab_set)

            prob_rows = []
            if smoothing.startswith("MLE"):
                total_c = sum(candidates.values())
                for cand, count in candidates.items():
                    p = count / total_c if total_c > 0 else 0.0
                    prob_rows.append({"Next Word": cand, "Count C(w1, w2)": count, "Probability P(w|prompt)": p})
            else:
                # Laplace Add-1
                # Show top observed candidates + smoothed unseen sample
                denom = c_prev + v_size
                for cand in vocab_set:
                    c_pair = candidates.get(cand, 0)
                    p = (c_pair + 1) / denom
                    if c_pair > 0:
                        prob_rows.append({"Next Word": cand, "Count C(w1, w2)": c_pair, "Probability P(w|prompt)": p})

            df_lm = pd.DataFrame(prob_rows)
            if not df_lm.empty:
                df_lm = df_lm.sort_values(by="Probability P(w|prompt)", ascending=False).reset_index(drop=True)
                
                with sel_col2:
                    st.markdown(f"**Top Candidates following:** `{selected_prompt}`")
                    top_candidate = df_lm.iloc[0]["Next Word"]
                    top_prob = df_lm.iloc[0]["Probability P(w|prompt)"]
                    st.info(f"Most likely next word: **'{top_candidate}'** (P = **{top_prob:.3f}** or **{top_prob*100:.1f}%**)")

                # Horizontal bar chart of next word candidates
                top_plot_df = df_lm.head(8)
                fig_lm = px.bar(
                    top_plot_df,
                    x="Probability P(w|prompt)",
                    y="Next Word",
                    orientation="h",
                    text=top_plot_df["Probability P(w|prompt)"].apply(lambda p: f"{p:.3f}"),
                    title=f"Next Word Conditional Probability Distribution: P( w | '{selected_prompt}' )",
                    color="Probability P(w|prompt)",
                    color_continuous_scale="Purples",
                )
                fig_lm.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor=COLOR_BG,
                    paper_bgcolor=COLOR_BG,
                )
                st.plotly_chart(fig_lm, use_container_width=True)

                # Interactive Markov Text Generator
                st.markdown("<b>🎲 Live Markov Next-Word Text Generator:</b>", unsafe_allow_html=True)
                gen_col1, gen_col2 = st.columns([1, 2])
                with gen_col1:
                    gen_steps = st.slider("Words to Generate:", min_value=2, max_value=8, value=4, key="lm_gen_slider")
                    gen_mode = st.radio("Sampling Strategy:", ["Greedy (Highest Probability)", "Stochastic (Weighted Probability)"], index=0)
                    btn_generate = st.button("▶️ Generate Continuation", key="btn_run_markov_gen")

                with gen_col2:
                    if btn_generate:
                        curr_word = selected_prompt
                        generated_seq = [curr_word]
                        for _ in range(gen_steps):
                            next_candidates = bigram_counts[curr_word]
                            if not next_candidates:
                                break
                            words_list = list(next_candidates.keys())
                            counts_list = list(next_candidates.values())
                            probs_list = np.array(counts_list) / sum(counts_list)
                            
                            if "Greedy" in gen_mode:
                                next_w = words_list[int(np.argmax(probs_list))]
                            else:
                                next_w = np.random.choice(words_list, p=probs_list)
                            generated_seq.append(next_w)
                            curr_word = next_w

                        gen_text = " ".join(generated_seq)
                        st.markdown(
                            f"""
                            <div style="background-color: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 8px; padding: 12px 16px; margin-top: 8px;">
                                <div style="font-weight: 700; color: {COLOR_PRIMARY}; font-size: 0.85rem; text-transform: uppercase;">Generated Markov Chain:</div>
                                <div style="font-size: 1.05rem; color: {COLOR_TEXT}; margin-top: 4px; font-weight: 500;">
                                    "{gen_text}..."
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption("Click 'Generate Continuation' to produce an autonomous sequence based on empirical text probabilities.")
            else:
                st.info(f"No observed transitions following '{selected_prompt}' in the input corpus.")
        else:
            # Trigram (N=3)
            trigram_counts = defaultdict(Counter)
            bigram_prefix_counts = Counter()
            for w1, w2, w3 in zip(lm_words[:-2], lm_words[1:-1], lm_words[2:]):
                trigram_counts[(w1, w2)][w3] += 1
                bigram_prefix_counts[(w1, w2)] += 1

            if trigram_counts:
                prefixes = [f"{w1} {w2}" for (w1, w2) in trigram_counts.keys()]
                sel_prefix = st.selectbox("Select Trigram Prefix (w1, w2):", prefixes)
                p1, p2 = sel_prefix.split(" ", 1)
                tri_cands = trigram_counts[(p1, p2)]
                total_tri = sum(tri_cands.values())

                tri_rows = [
                    {"Next Word (w3)": w3, "Count C(w1, w2, w3)": c, "Probability P(w3|w1, w2)": c / total_tri}
                    for w3, c in tri_cands.items()
                ]
                df_tri = pd.DataFrame(tri_rows).sort_values(by="Probability P(w3|w1, w2)", ascending=False)
                st.dataframe(df_tri, use_container_width=True, hide_index=True)
            else:
                st.info("Input text needs at least 3 consecutive words to form trigrams.")
    else:
        st.info("Enter at least 4 words in the text area above to explore N-gram Language Modeling.")

    # 9. Structured Table of Tokens
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Token Breakdown Table")
    df = pd.DataFrame(token_details)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export Token Table Button
    st.download_button(
        label="📥 Download Token Breakdown (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="token_breakdown.csv",
        mime="text/csv",
        key="btn_download_token_table"
    )

    # 10. Educational Takeaway / Comparison Card
    with st.expander("💡 Deep Dive: How Different Tokenizers Treat Contractions & Symbols"):
        st.markdown(
            """
            * **Contractions:** Compare how words like `Don't` or `it's` are split.
              * Standard Python `.split()` gives: `["Don't", "you", "think", "it's"]` (keeps punctuation glued to words).
              * NLTK / spaCy tokenizers give: `["Do", "n't"]` and `["it", "'s"]`, separating the auxiliary verb from the negative particle `not` so downstream sentiment and grammar models understand both!
            * **Punctuation:** Real NLP tokenizers peel commas, periods, and quotation marks into separate tokens rather than treating `"seconds!"` as a distinct word from `"seconds"`.
            """
        )

    # 11. Optional Bonus: Marathi / Indic Script Tokenization & Akshara Morphology (Task 6 Item 7)
    with st.expander("🇮🇳 Optional Bonus: Marathi / Hindi (Devanagari) Tokenization & Akshara Segmentation"):
        st.markdown(
            "Explore how NLP handles non-Latin scripts like **Marathi** and **Hindi** where compound characters (जोडाक्षरे), "
            "vowel matras (मात्रा), and danda punctuation (`।`) require Unicode-aware segmentation:"
        )
        default_marathi = "नैसर्गिक भाषा प्रक्रिया (NLP) संगणकाला मानवी भाषा समजण्यास मदत करते. तुम्ही अभ्यास करत आहात का?"
        indic_text = st.text_input("Enter Marathi / Hindi text:", value=default_marathi, key="indic_token_input")
        if indic_text.strip():
            import re
            # 1. Word tokens
            indic_tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+|[^\s\w]', indic_text, re.UNICODE)
            indic_chips = []
            for idx, itok in enumerate(indic_tokens):
                color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
                indic_chips.append(chip(itok, color=color, subtitle=f"#{idx+1}"))
            st.markdown("**1. Word-Level Tokens:**")
            st.markdown(f'<div class="chip-container">{" ".join(indic_chips)}</div>', unsafe_allow_html=True)

            # 2. Akshara (Orthographic Syllable) Segmentation
            # Devanagari Akshara: independent vowel OR (consonant + virama)* + consonant + [matra] + [modifier]
            akshara_regex = re.compile(
                r'(?:[\u0904-\u0914\u0960\u0961]|(?:[\u0915-\u0939\u0958-\u095F]\u094D)*[\u0915-\u0939\u0958-\u095F][\u093E-\u094C\u094E\u0955-\u0957\u0962\u0963]?)(?:[\u0901-\u0903\u093C])?'
            )
            dev_words = [w for w in indic_tokens if re.match(r'[\u0900-\u097F]', w)]
            all_aksharas = []
            conjunct_count = 0
            matra_count = 0

            for dw in dev_words:
                found = akshara_regex.findall(dw)
                all_aksharas.extend(found)
                # Count conjuncts (virama \u094D)
                conjunct_count += dw.count('\u094D')
                # Count matras
                matra_count += len(re.findall(r'[\u093E-\u094C\u094E\u0955-\u0957\u0962\u0963]', dw))

            st.markdown("<br><b>2. Akshara (अक्षर / Syllable) Segmentation:</b>", unsafe_allow_html=True)
            akshara_chips = [
                chip(ak, color=PASTEL_COLORS[i % len(PASTEL_COLORS)], subtitle="Akshara")
                for i, ak in enumerate(all_aksharas[:24])
            ]
            st.markdown(f'<div class="chip-container">{" ".join(akshara_chips)}</div>', unsafe_allow_html=True)

            c_ind1, c_ind2, c_ind3, c_ind4 = st.columns(4)
            with c_ind1:
                st.metric("Total Words", len(indic_tokens))
            with c_ind2:
                st.metric("Total Aksharas (अक्षरे)", len(all_aksharas))
            with c_ind3:
                st.metric("Vowel Matras (मात्रा)", matra_count)
            with c_ind4:
                st.metric("Virama / Conjuncts (जोडाक्षरे)", conjunct_count)

