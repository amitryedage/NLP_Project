"""
Cached NLP resource and model loaders for NLP Visual Lab.
Utilizes @st.cache_resource to load heavy models once across sessions.
"""
import streamlit as st
import nltk

# Essential NLTK packages required across modules
NLTK_PACKAGES = ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]


@st.cache_resource(show_spinner=False)
def ensure_nltk_data():
    """
    Downloads required NLTK corpora once in a safe, non-blocking manner.
    """
    downloaded = []
    for pkg in NLTK_PACKAGES:
        try:
            # Check if resource is already available locally
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
            downloaded.append(pkg)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
                downloaded.append(pkg)
            except Exception as e:
                # Log or allow graceful fallback
                print(f"Warning: Failed to download NLTK package '{pkg}': {e}")
    return downloaded


@st.cache_resource(show_spinner="Loading spaCy model ('en_core_web_md')...")
def load_spacy_model():
    """
    Loads and caches the spaCy medium English model.
    """
    import spacy
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        # Fallback if package is registered under canonical wheel name
        try:
            import en_core_web_md
            nlp = en_core_web_md.load()
        except Exception as e:
            st.error(
                "spaCy model `en_core_web_md` is not installed. "
                "Please run `uv sync` to install dependencies."
            )
            raise e
    return nlp


@st.cache_resource(show_spinner=False)
def get_stemmers_and_lemmatizers():
    """
    Returns cached instances of Porter, Snowball stemmers and WordNet lemmatizer.
    """
    ensure_nltk_data()
    from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
    
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    
    return {
        "porter": porter,
        "snowball": snowball,
        "lemmatizer": lemmatizer
    }
