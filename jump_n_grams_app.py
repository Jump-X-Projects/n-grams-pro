# jump_n_grams_app.py

import streamlit as st
import re
import io
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import base64

# ------------------------------------------------------
# 1. HELPER FUNCTIONS (DATA LOADING, TEXT PREPROCESSING)
# ------------------------------------------------------

def load_text_file(uploaded_file):
    """
    Loads text from an uploaded file.
    Returns a string of the text content.
    """
    try:
        bytes_data = uploaded_file.read()
        text_data = bytes_data.decode("utf-8", errors="ignore")
        return text_data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def preprocess_text(text, lowercase=True, remove_punct=True):
    """
    Basic text preprocessing.
    - Optionally lowercases text.
    - Optionally removes punctuation.
    Returns a list of tokens.
    """
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return tokens


# ------------------------------------------------------
# 2. CORE LOGIC: BUILDING JUMP-N-GRAMS
# ------------------------------------------------------

def get_jump_ngrams(tokens, n=2, jump=1):
    """
    Computes jump-n-grams (skip-grams) from a list of tokens.
    - n: n-gram size (e.g., 2 for bigrams, 3 for trigrams).
    - jump: number of tokens to skip between each word in the n-gram.
      For example, if n=2 and jump=1, you look at pairs separated by 1 token.
      If jump=0, it becomes a standard n-gram (adjacent tokens).

    Returns a list of n-gram tuples.
    """
    if n < 1:
        # Basic validation
        st.warning("n-gram size must be at least 1. Using default of 2.")
        n = 2

    jump_ngrams_list = []
    # For each starting index in the tokens list
    for i in range(len(tokens) - (n - 1) * (jump + 1)):
        # Build the n-gram by skipping 'jump' tokens between each piece
        gram_tokens = []
        for j in range(n):
            gram_tokens.append(tokens[i + j*(jump+1)])
        jump_ngrams_list.append(tuple(gram_tokens))
    return jump_ngrams_list

@st.cache_data(show_spinner=False)
def compute_jump_ngrams(tokens, n, jump):
    """
    Cached function: given a list of tokens, n, and jump,
    returns a DataFrame of the frequency distribution of jump-n-grams.
    """
    jngrams = get_jump_ngrams(tokens, n=n, jump=jump)
    counter = Counter(jngrams)
    df = pd.DataFrame(counter.items(), columns=["Jump-n-gram", "Frequency"])
    df.sort_values(by="Frequency", ascending=False, inplace=True)
    return df


# ------------------------------------------------------
# 3. VISUALIZATION FUNCTIONS
# ------------------------------------------------------

def plot_top_jump_ngrams(df, top_k=20):
    """
    Plots a bar chart of the top-k jump-n-grams from the provided DataFrame.
    """
    df_top = df.head(top_k)
    plt.figure(figsize=(10, 6))
    plt.barh(
        [str(gram) for gram in df_top["Jump-n-gram"]],
        df_top["Frequency"],
        color="skyblue"
    )
    plt.gca().invert_yaxis()  # highest freq at top
    plt.xlabel("Frequency")
    plt.title(f"Top {top_k} Jump-n-grams")
    st.pyplot(plt.gcf())


# ------------------------------------------------------
# 4. MAIN APP LAYOUT
# ------------------------------------------------------

def main():
    st.title("Jump-n-Grams Tool")
    st.markdown(
        """
        This app computes **jump-n-grams** (skip-grams) from a given text.
        - **n**: size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).
        - **jump**: number of tokens to skip between words in the n-gram.

        **Example**: If n=2 and jump=1, you capture pairs separated by 1 token.
        """
    )

    # Sidebar for parameters
    st.sidebar.header("Jump-n-Gram Parameters")
    n_val = st.sidebar.slider("N-gram size (n)", min_value=1, max_value=5, value=2)
    jump_val = st.sidebar.slider("Jump size", min_value=0, max_value=5, value=0)
    top_k_val = st.sidebar.slider("How many top results to visualize?", 5, 50, 20)

    # -------------- Data Input Section --------------
    st.subheader("1. Upload or Load Sample Data")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "csv", "md"])

    # We also provide a sample text
    sample_text = """This is a sample text to demonstrate jump-n-grams or skip-grams.
    Jump-n-grams allow you to capture context in a text where the words may not be adjacent.
    For example, if we skip 1 word between pairs, we get interesting patterns."""

    # Let users load sample data
    use_sample = st.checkbox("Use sample text instead of uploading")

    if uploaded_file is None and not use_sample:
        st.info("Upload a file or check the 'Use sample text' box to proceed.")
        return

    # If user uses sample or uploads a file, load the text
    if use_sample:
        text_data = sample_text
    else:
        text_data = load_text_file(uploaded_file)

    # -------------- Preprocessing Options --------------
    st.subheader("2. Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        lowercase_opt = st.checkbox("Lowercase text", value=True)
    with col2:
        remove_punct_opt = st.checkbox("Remove punctuation", value=True)

    # -------------- Generate Jump-n-Grams --------------
    if st.button("Compute Jump-n-grams"):
        with st.spinner("Computing jump-n-grams..."):
            tokens = preprocess_text(text_data, lowercase=lowercase_opt, remove_punct=remove_punct_opt)
            jngram_df = compute_jump_ngrams(tokens, n=n_val, jump=jump_val)

            st.subheader("Results")
            st.write(f"Total unique jump-n-grams found: **{len(jngram_df)}**")

            st.dataframe(jngram_df.head(top_k_val))

            # Show top k chart
            plot_top_jump_ngrams(jngram_df, top_k=top_k_val)

            # Download results button
            csv_data = jngram_df.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="jump_ngrams_results.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("Click 'Compute Jump-n-grams' to analyze your text.")


if __name__ == "__main__":
    main()
