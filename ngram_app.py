import streamlit as st
import pandas as pd
import nltk
import logging
from google_ads_connector import GoogleAdsConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from preprocessing import preprocess_dataframe
from analysis import generate_ngrams, find_collocations, compute_tfidf, compute_ads_metrics
from visualization import (
    create_word_cloud,
    plot_dual_axis_chart,
    plot_color_coded_bars,
    plot_side_by_side_bars,
    plot_heatmap,
    plot_bubble_chart,
    plot_ngram_analysis
)

st.set_page_config(
    page_title="N-Gram Analysis Tool (Pro)",
    page_icon="📊",
    layout="wide"
)

def validate_numeric_columns(df, cost_col, conversions_col):
    """Validate and clean numeric columns"""
    try:
        # Convert cost column (remove $ and ,) - using raw string for escape sequence
        df[cost_col] = df[cost_col].replace(r'[$,]', '', regex=True).astype(float)

        # Convert conversions column
        df[conversions_col] = pd.to_numeric(df[conversions_col], errors='coerce')

        # Check for NaN values
        if df[cost_col].isna().any() or df[conversions_col].isna().any():
            st.warning("Some numeric values could not be converted. They will be treated as 0.")
            df[cost_col] = df[cost_col].fillna(0)
            df[conversions_col] = df[conversions_col].fillna(0)

        return df
    except Exception as e:
        logger.error(f"Error converting numeric columns: {str(e)}")
        st.error(f"Error converting numeric columns: {str(e)}")
        return None

def main():
    st.title("N-Gram Analysis Tool (Pro Version)")
    st.write(
        "Analyze search terms and performance metrics from CSV or directly from Google Ads. "
        "Follow the steps below to configure your analysis."
    )

    # Data Source Selection
    data_source = st.radio(
        "Choose Data Source",
        ["Upload CSV", "Google Ads Data"],
        help="Select where to get your data from"
    )

    if data_source == "Google Ads Data":
        with st.expander("Google Ads Configuration", expanded=True):
            customer_id = st.text_input(
                "Customer ID (without dashes)",
                help="Your Google Ads account ID"
            )

            date_range = st.selectbox(
                "Date Range",
                ["LAST_30_DAYS", "LAST_7_DAYS", "LAST_14_DAYS", "LAST_90_DAYS"],
                help="Select the time period for search terms"
            )

            if st.button("Fetch Google Ads Data"):
                with st.spinner("Fetching data from Google Ads..."):
                    connector = GoogleAdsConnector()
                    df = connector.get_search_terms_report(customer_id, date_range)

                    if df is not None:
                        st.session_state['df'] = df
                        st.session_state['search_term_col'] = 'search_term'
                        st.session_state['cost_col'] = 'cost'
                        st.session_state['conversions_col'] = 'conversions'
                        st.success("Data fetched successfully!")

                        # Show data preview
                        st.write("Data Preview:")
                        st.dataframe(df.head())
                    else:
                        st.error("Failed to fetch data from Google Ads")

    else:
        # CSV upload section
        st.subheader("Upload Your CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                # Column selection
                st.subheader("Select Your Columns")
                cols = list(df.columns)

                search_term_col = st.selectbox("Search Terms Column", cols)
                cost_col = st.selectbox("Cost Column", cols)
                conversions_col = st.selectbox("Conversions Column", cols)

                # Store in session state
                st.session_state['df'] = df
                st.session_state['search_term_col'] = search_term_col
                st.session_state['cost_col'] = cost_col
                st.session_state['conversions_col'] = conversions_col

            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
                return

    # Analysis Configuration
    if 'df' in st.session_state:
        st.subheader("Configure Analysis")

        # Preprocessing options
        with st.expander("Preprocessing Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                remove_stop = st.checkbox("Remove Stopwords", value=True)
                remove_punc = st.checkbox("Remove Punctuation", value=True)
            with col2:
                to_lower = st.checkbox("Lowercase", value=True)
                custom_sw = st.text_area(
                    "Custom Stopwords (comma-separated)",
                    help="Enter additional words to remove, separated by commas"
                )

        custom_sw_list = [w.strip() for w in custom_sw.split(",") if w.strip()]

        # N-gram configuration
        with st.expander("N-gram Configuration", expanded=True):
            col3, col4 = st.columns(2)
            with col3:
                n_value = st.selectbox(
                    "N-gram size",
                    options=[1, 2, 3, 4, 5],
                    index=1
                )
            with col4:
                freq_threshold = st.slider(
                    "Minimum Frequency",
                    min_value=1,
                    max_value=50,
                    value=2
                )

        # Process Data
        if st.button("Process Data", type="primary"):
            with st.spinner("Processing data..."):
                try:
                    df = st.session_state['df']

                    # Validate numeric columns
                    df = validate_numeric_columns(
                        df,
                        st.session_state['cost_col'],
                        st.session_state['conversions_col']
                    )

                    if df is None:
                        return

                    # Preprocess text
                    df_processed = preprocess_dataframe(
                        df,
                        st.session_state['search_term_col'],
                        remove_stopwords=remove_stop,
                        remove_punctuation=remove_punc,
                        lowercase=to_lower,
                        custom_stopwords=custom_sw_list
                    )

                    # Generate n-grams
                    df_ngrams = generate_ngrams(
                        df_processed['processed_text'].tolist(),
                        n=n_value,
                        freq_threshold=freq_threshold
                    )

                    # Calculate metrics
                    df_metrics = compute_ads_metrics(
                        df,
                        df_ngrams,
                        st.session_state['search_term_col'],
                        st.session_state['cost_col'],
                        st.session_state['conversions_col']
                    )

                    # Display results
                    st.success("Analysis complete!")

                    # Show metrics table and visualizations using plot_ngram_analysis
                    plot_ngram_analysis(
                        df_ngrams=df_ngrams,
                        df_original=df,
                        search_term_col=st.session_state['search_term_col'],
                        cost_col=st.session_state['cost_col'],
                        conversions_col=st.session_state['conversions_col']
                    )

                    # Word cloud for unigrams and bigrams
                    if n_value <= 2:
                        st.subheader("Word Cloud Visualization")
                        create_word_cloud(df_ngrams)

                    # Download results
                    csv = df_metrics.to_csv(index=False)
                    st.download_button(
                        "Download Results (CSV)",
                        csv,
                        "ngram_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    logger.error(f"Processing error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
