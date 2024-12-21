import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt

def plot_ngram_bar_chart(df_ngrams: pd.DataFrame, top_k: int = 20):
    """
    Render a bar chart of the most frequent n-grams.
    :param df_ngrams: DataFrame with ['ngram', 'frequency'] columns.
    :param top_k: Number of top n-grams to display.
    """
    df_plot = df_ngrams.head(top_k).copy()

    # Updated Altair chart configuration
    chart = (
        alt.Chart(df_plot)
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
