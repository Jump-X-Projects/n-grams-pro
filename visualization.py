import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt

def plot_ngram_analysis(df_ngrams: pd.DataFrame, df_original: pd.DataFrame, search_term_col: str, cost_col: str, conversions_col: str, top_k: int = 20):
    """
    Display a table of n-gram statistics and a bar chart.
    :param df_ngrams: DataFrame with ['ngram', 'frequency'] columns
    :param df_original: Original DataFrame with cost and conversion data
    :param search_term_col: Name of search term column
    :param cost_col: Name of cost column
    :param conversions_col: Name of conversions column
    :param top_k: Number of top n-grams to display
    """
    # Get top k n-grams
    df_top = df_ngrams.head(top_k).copy()

    # Calculate metrics for each n-gram
    metrics = []
    for ngram in df_top['ngram']:
        # Find search terms containing this n-gram
        mask = df_original[search_term_col].str.contains(ngram, case=False, na=False)
        total_cost = df_original.loc[mask, cost_col].sum()
        total_conversions = df_original.loc[mask, conversions_col].sum()
        cpa = total_cost / total_conversions if total_conversions > 0 else 0

        metrics.append({
            'N-gram': ngram,
            'Frequency': df_top[df_top['ngram'] == ngram]['frequency'].iloc[0],
            'Total Cost': total_cost,
            'Total Conversions': total_conversions,
            'CPA': cpa
        })

    # Create metrics DataFrame
    df_metrics = pd.DataFrame(metrics)

    # Format currency columns
    df_metrics['Total Cost'] = df_metrics['Total Cost'].apply(lambda x: f"${x:,.2f}")
    df_metrics['CPA'] = df_metrics['CPA'].apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")

    # Display table
    st.dataframe(
        df_metrics,
        column_config={
            'N-gram': st.column_config.TextColumn("N-gram"),
            'Frequency': st.column_config.NumberColumn("Frequency"),
            'Total Cost': st.column_config.TextColumn("Total Cost"),
            'Total Conversions': st.column_config.NumberColumn("Total Conversions"),
            'CPA': st.column_config.TextColumn("CPA")
        },
        hide_index=True
    )

    # Create and display bar chart
    chart = (
        alt.Chart(df_top)
        .mark_bar()
        .encode(
            x=alt.X('frequency:Q', title='Frequency'),
            y=alt.Y('ngram:N', sort='-x', title='N-gram'),
            tooltip=['ngram:N', 'frequency:Q']
        )
        .properties(
            title=f"Top {top_k} N-grams",
            height=min(top_k * 25, 500)
        )
    )
    st.altair_chart(chart, use_container_width=True)

def create_word_cloud(df_ngrams: pd.DataFrame, max_words: int = 100):
    """
    Create and display a word cloud from a DataFrame of n-grams.
    Note: This is typically more meaningful for unigrams or bigrams only.
    :param df_ngrams: DataFrame with ['ngram', 'frequency'] columns.
    :param max_words: The maximum number of words in the word cloud.
    """
    if df_ngrams.empty:
        st.warning("No data available for word cloud visualization")
        return

    # Convert the frequency table to a dict
    freq_dict = dict(zip(df_ngrams['ngram'], df_ngrams['frequency']))

    # Updated WordCloud configuration
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        normalize_plurals=False,
        collocations=False
    )

    try:
        wc.generate_from_frequencies(freq_dict)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"Could not generate word cloud: {str(e)}")
