import streamlit as st
import pandas as pd
import gc
import logging
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

    ########
