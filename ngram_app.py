import streamlit as st
import pandas as pd
import nltk
import logging

# Download required NLTK data at startup
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

from preprocessing import preprocess_dataframe
from analysis import generate_ngrams, find_collocations, compute_tfidf
from visualization import (
    create_word_cloud,
    plot_dual_axis_chart,
    plot_color_coded_bars,
    plot_side_by_side_bars,
    plot_heatmap,
    plot_bubble_chart
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="N-Gram Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS to handle blank screen issues
st.markdown("""
    <style>
        .main {
            padding: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def get_csv_columns(file):
    """Get column names from CSV without loading the entire file."""
    try:
        # Read just the header row
        df_header = pd.read_csv(file, nrows=0)
        return list(df_header.columns)
    except Exception as e:
        logger.error(f"Error reading CSV headers: {str(e)}")
        st.error(f"Error reading CSV headers: {str(e)}")
        return []
    finally:
        # Reset file pointer to beginning
        file.seek(0)

def handle_empty_data(df):
    """Check if DataFrame is empty and handle appropriately"""
    if df.empty:
        st.error("No data found in the uploaded file.")
        return True
    return False

def validate_numeric_columns(df, cost_col, conversions_col):
    """Validate and clean numeric columns"""
    try:
        # Convert cost column (remove $ and ,)
        df[cost_col] = df[cost_col].replace('[\$,]', '', regex=True).astype(float)

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
    try:
        st.title("N-Gram Analysis Tool")
        st.write(
            "Upload a CSV file to analyze search terms and their performance metrics. "
            "Follow the steps below to configure your analysis."
        )

        ########################################################
        # Step 1: File Upload
        ########################################################
        st.subheader("Step 1: Upload Your CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if not uploaded_file:
            st.info("Please upload a CSV file to begin.")
            return

        # Get column names after file upload
        columns = get_csv_columns(uploaded_file)
        if not columns:
            st.error("Could not read column names from the CSV file.")
            return

        ########################################################
        # Step 2: Column Selection
        ########################################################
        st.subheader("Step 2: Select Your Columns")

        col1, col2 = st.columns(2)

        with col1:
            search_term_col = st.selectbox(
                "Search Terms Column",
                options=columns,
                help="Select the column containing your search terms"
            )

            cost_col = st.selectbox(
                "Cost Column",
                options=columns,
                help="Select the column containing cost data"
            )

        with col2:
            conversions_col = st.selectbox(
                "Conversions Column",
                options=columns,
                help="Select the column containing conversion data"
            )

        ########################################################
        # Step 3: Preprocessing Options
        ########################################################
        st.subheader("Step 3: Configure Preprocessing")

        col3, col4 = st.columns(2)

        with col3:
            remove_stop = st.checkbox("Remove Stopwords", value=True)
            remove_punc = st.checkbox("Remove Punctuation", value=True)

        with col4:
            to_lower = st.checkbox("Lowercase", value=True)
            custom_sw = st.text_area(
                "Custom Stopwords (comma-separated)",
                placeholder="Enter any additional words to remove, separated by commas"
            )

        custom_sw_list = [w.strip() for w in custom_sw.split(",") if w.strip()]

        ########################################################
        # Step 4: Analysis Configuration
        ########################################################
        st.subheader("Step 4: Configure Analysis")

        col5, col6 = st.columns(2)

        with col5:
            n_value = st.selectbox(
                "N-gram size",
                options=[1, 2, 3, 4, 5],
                index=1,
                help="Size of n-grams to generate (1=unigrams, 2=bigrams, etc.)"
            )

        with col6:
            freq_threshold = st.slider(
                "Minimum Frequency",
                min_value=1,
                max_value=50,
                value=2,
                help="Exclude n-grams appearing less than this many times"
            )

        ########################################################
        # Step 5: Process Button
        ########################################################
        st.subheader("Step 5: Run Analysis")

        if st.button("Process Data", type="primary"):
            with st.spinner("Processing your data..."):
                try:
                    # Read the CSV with better error handling
                    df = pd.read_csv(uploaded_file, on_bad_lines='warn')
                    if handle_empty_data(df):
                        return

                    # Comprehensive validation
                    required_columns = [search_term_col, cost_col, conversions_col]

                    # Check for duplicate column selections
                    if len(set(required_columns)) != len(required_columns):
                        st.error("Please select different columns for each field.")
                        return

                    # Validate all required columns exist
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        return

                    # Validate data in columns
                    df = validate_numeric_columns(df, cost_col, conversions_col)
                    if df is None:
                        return

                    # Check for empty text column
                    if df[search_term_col].isna().all() or df[search_term_col].str.strip().eq('').all():
                        st.error("Search terms column is empty!")
                        return

                    # Preprocess the search terms
                    df_processed = preprocess_dataframe(
                        df,
                        text_column=search_term_col,
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

                    # Display results
                    st.success("Analysis complete!")

                    # Calculate metrics
                    metrics = []
                    for ngram in df_ngrams.head(20)['ngram']:
                        mask = df[search_term_col].astype(str).str.contains(
                            str(ngram), case=False, na=False
                        )
                        total_cost = df.loc[mask, cost_col].sum()
                        total_conversions = df.loc[mask, conversions_col].sum()
                        cpa = total_cost / total_conversions if total_conversions > 0 else 0

                        metrics.append({
                            'N-gram': str(ngram),
                            'Frequency': int(df_ngrams[df_ngrams['ngram'] == ngram]['frequency'].iloc[0]),
                            'Total Cost': float(total_cost),
                            'Total Conversions': int(total_conversions),
                            'CPA': float(cpa),
                            'CPA_float': float(cpa),
                            'Total_Cost_float': float(total_cost)
                        })

                    df_metrics = pd.DataFrame(metrics)

                    # Format currency columns for display
                    df_metrics_display = df_metrics.copy()
                    df_metrics_display['Total Cost'] = df_metrics_display['Total Cost'].apply(lambda x: f"${x:,.2f}")
                    df_metrics_display['CPA'] = df_metrics_display['CPA'].apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")

                    # Display metrics table
                    st.dataframe(
                        df_metrics_display[['N-gram', 'Frequency', 'Total Cost', 'Total Conversions', 'CPA']],
                        column_config={
                            'N-gram': st.column_config.TextColumn("N-gram"),
                            'Frequency': st.column_config.NumberColumn("Frequency"),
                            'Total Cost': st.column_config.TextColumn("Total Cost"),
                            'Total Conversions': st.column_config.NumberColumn("Total Conversions"),
                            'CPA': st.column_config.TextColumn("CPA")
                        },
                        hide_index=True
                    )

                    # Display visualizations
                    st.subheader("1. Dual Axis Bar/Line Chart")
                    plot_dual_axis_chart(df_metrics)

                    st.subheader("2. Color-Coded Bars (by CPA)")
                    plot_color_coded_bars(df_metrics)

                    st.subheader("3. Side-by-Side Bars")
                    plot_side_by_side_bars(df_metrics)

                    st.subheader("4. Heat Map")
                    plot_heatmap(df_metrics)

                    st.subheader("5. Bubble Chart")
                    plot_bubble_chart(df_metrics)

                    # Optional word cloud for unigrams and bigrams
                    if n_value <= 2:
                        st.subheader("Word Cloud Visualization")
                        create_word_cloud(df_ngrams)

                    # Allow download of results
                    csv_data = df_metrics_display.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="ngram_analysis_results.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    logger.error(f"Error during processing: {str(e)}")
                    st.error("An error occurred during processing. Please check your data and try again.")
                    st.error(f"Error details: {str(e)}")

    except Exception as e:
        logger.error(f"Main app error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()
