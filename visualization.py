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
    df_plot = df_ngrams.head(top_k)
    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X('frequency:Q', sort='-y'),
            y=alt.Y('ngram:N', sort=None),
            tooltip=['frequency']
        )
        .properties(title=f"Top {top_k} N-grams")
    )
    st.altair_chart(chart, use_container_width=True)

def create_word_cloud(df_ngrams: pd.DataFrame, max_words: int = 100):
    """
    Create and display a word cloud from a DataFrame of n-grams.
    Note: This is typically more meaningful for unigrams or bigrams only.
    :param df_ngrams: DataFrame with ['ngram', 'frequency'] columns.
    :param max_words: The maximum number of words in the word cloud.
    """
    # Convert the frequency table to a dict
    freq_dict = dict(zip(df_ngrams['ngram'], df_ngrams['frequency']))

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words
    )
    wc.generate_from_frequencies(freq_dict)

    # Display with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
