import streamlit as st
import pandas as pd
from preprocessing import preprocess_dataframe
from analysis import generate_ngrams, find_collocations, compute_tfidf
from visualization import create_word_cloud
from visualization import plot_dual_axis_chart
from visualization import plot_color_coded_bars
from visualization import plot_side_by_side_bars
from visualization import plot_heatmap
from visualization import plot_bubble_chart

def get_csv_columns(file):
    """Get column names from CSV without loading the entire file."""
    try:
        # Read just the header row
        df_header = pd.read_csv(file, nrows=0)
        return list(df_header.columns)
    except Exception as e:
        st.error(f"Error reading CSV headers: {str(e)}")
        return []
    finally:
        # Reset file pointer to beginning
        file.seek(0)

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
                # Read the CSV
                df = pd.read_csv(uploaded_file)

                # Basic validation
                required_columns = [search_term_col, cost_col, conversions_col]
                if len(set(required_columns)) != len(required_columns):
                    st.error("Please select different columns for each field.")
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
                    mask = df[search_term_col].astype(str).str.contains(str(ngram), case=False, na=False)
                    total_cost = df.loc[mask, cost_col].apply(lambda x: float(str(x).replace('$', '').replace(',', ''))).sum()
                    total_conversions = df.loc[mask, conversions_col].astype(float).sum()
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

                # Display all visualization options
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

                # Optional visualizations
                if n_value <= 2:  # Word cloud only makes sense for unigrams and bigrams
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
                st.error(f"Error during processing: {str(e)}")
                st.error("Please check your column selections and try again.")

if __name__ == "__main__":
    main()
