import streamlit as st
import pandas as pd
from preprocessing import load_csv_in_chunks, preprocess_dataframe
from analysis import generate_ngrams, find_collocations, compute_tfidf
from visualization import plot_ngram_bar_chart, create_word_cloud

# Optional if using spaCy for advanced POS filtering
# import spacy
# nlp = spacy.load("en_core_web_sm")

########################################################
# Streamlit App
########################################################

def main():
    st.title("N-Gram Analysis Tool")
    st.write(
        "Upload a CSV of up to ~1 million rows or paste your own text. "
        "Configure preprocessing and generate n-grams, collocations, and TF-IDF."
    )

    ########################################################
    # Sidebar for configuration
    ########################################################
    st.sidebar.header("Configuration")

    data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Paste Text"])

    remove_stop = st.sidebar.checkbox("Remove Stopwords", value=True)
    remove_punc = st.sidebar.checkbox("Remove Punctuation", value=True)
    to_lower = st.sidebar.checkbox("Lowercase", value=True)

    custom_sw = st.sidebar.text_area("Custom Stopwords (comma-separated)", "")
    custom_sw_list = [w.strip() for w in custom_sw.split(",") if w.strip()]

    ########################################################
    # Step 1: Load Data
    ########################################################

    df = pd.DataFrame()
    text_column = "text"  # default column name if uploading CSV

    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Ask user for the column containing text
            text_column = st.sidebar.text_input("Text Column Name", value="text")
            st.write("Loading data... (may take a moment for large files)")
            # Load in chunks to handle up to ~1m rows
            df = load_csv_in_chunks(
                file=uploaded_file,
                text_column=text_column,
                chunksize=100000,
                max_rows=1000000
            )
            st.write(f"Loaded {len(df)} rows from CSV.")

    else:  # Paste Text
        raw_text = st.sidebar.text_area("Paste your text here")
        if raw_text:
            df = pd.DataFrame({text_column: [raw_text]})
            st.write("Using pasted text.")

    if df.empty:
        st.warning("No data loaded yet. Please upload a CSV or paste text.")
        return

    ########################################################
    # Step 2: Preprocess Data
    ########################################################
    st.write("Preprocessing text...")
    df = preprocess_dataframe(
        df,
        text_column=text_column,
        remove_stopwords=remove_stop,
        remove_punctuation=remove_punc,
        lowercase=to_lower,
        custom_stopwords=custom_sw_list
    )

    # We now have a 'processed_text' column
    st.success("Preprocessing complete!")

    ########################################################
    # Step 3: N-Gram Generation
    ########################################################
    st.subheader("N-Gram Analysis")
    n_value = st.selectbox("N-gram size", [1, 2, 3, 4, 5], index=0)
    freq_threshold = st.slider(
        "Frequency Threshold (exclude n-grams below this count)",
        min_value=1, max_value=50, value=1
    )

    if st.button("Generate N-Grams"):
        st.write("Generating N-Grams...")
        df_ngrams = generate_ngrams(
            df['processed_text'].tolist(),
            n=n_value,
            freq_threshold=freq_threshold
        )
        st.write(f"Found {len(df_ngrams)} n-grams (after filtering).")

        if not df_ngrams.empty:
            plot_ngram_bar_chart(df_ngrams)

            # Word Cloud (optional, more relevant for small n-values)
            if n_value <= 2:
                create_word_cloud(df_ngrams)

            # Allow download
            csv_data = df_ngrams.to_csv(index=False)
            st.download_button(
                label="Download N-Grams as CSV",
                data=csv_data,
                file_name="n_grams.csv",
                mime="text/csv"
            )
        else:
            st.warning("No n-grams to display with the current threshold.")

    ########################################################
    # Step 4: Collocation (optional)
    ########################################################
    st.subheader("Collocation Detection (Optional)")
    if st.checkbox("Run Bigram Collocation Detection?"):
        colloc_freq_threshold = st.slider("Collocation Frequency Filter", 2, 20, 2)
        colloc_top_n = st.slider("Show Top N Collocations", 5, 50, 20)

        if st.button("Find Collocations"):
            st.write("Finding collocations...")
            collocations = find_collocations(
                df['processed_text'].tolist(),
                freq_threshold=colloc_freq_threshold,
                top_n=colloc_top_n
            )
            if collocations:
                st.write("Top Collocations (word1, word2, PMI):")
                for (w1, w2), score in collocations:
                    st.write(f"**{w1} {w2}**: {score:.4f}")
            else:
                st.write("No collocations found.")

    ########################################################
    # Step 5: TF-IDF (optional)
    ########################################################
    st.subheader("TF-IDF Analysis (Optional)")
    if st.checkbox("Compute TF-IDF for top words?"):
        tfidf_top_n = st.slider("Top N Terms by TF-IDF", 5, 50, 20)

        if st.button("Compute TF-IDF"):
            st.write("Computing TF-IDF... (this may take a moment)")
            df_tfidf = compute_tfidf(df['processed_text'].tolist(), top_n=tfidf_top_n)
            st.write(df_tfidf)

            # Allow download
            csv_data = df_tfidf.to_csv(index=False)
            st.download_button(
                label="Download TF-IDF as CSV",
                data=csv_data,
                file_name="tfidf_scores.csv",
                mime="text/csv"
            )

    st.write("Done!")

if __name__ == "__main__":
    main()
