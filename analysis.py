import pandas as pd
import nltk
from nltk import ngrams
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Tuple, Dict

def compute_ads_metrics(
    df_original: pd.DataFrame,
    df_ngrams: pd.DataFrame,
    search_term_col: str,
    cost_col: str,
    conversions_col: str
) -> pd.DataFrame:
    """
    Compute advertising metrics for each n-gram.

    Parameters:
    -----------
    df_original : pandas.DataFrame
        Original DataFrame with search terms and metrics
    df_ngrams : pandas.DataFrame
        DataFrame with n-grams and their frequencies
    search_term_col : str
        Name of column containing search terms
    cost_col : str
        Name of column containing cost data
    conversions_col : str
        Name of column containing conversion data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with n-grams and their associated metrics
    """
    metrics = []

    for ngram in df_ngrams['ngram']:
        # Find search terms containing this n-gram
        mask = df_original[search_term_col].astype(str).str.contains(str(ngram), case=False, na=False)

        # Calculate metrics
        total_cost = df_original.loc[mask, cost_col].astype(float).sum()
        total_conversions = df_original.loc[mask, conversions_col].astype(float).sum()
        frequency = int(df_ngrams[df_ngrams['ngram'] == ngram]['frequency'].iloc[0])

        # Calculate CPA (Cost Per Acquisition)
        cpa = total_cost / total_conversions if total_conversions > 0 else 0

        metrics.append({
            'N-gram': ngram,
            'Frequency': frequency,
            'Total Cost': total_cost,
            'Total Conversions': total_conversions,
            'CPA': cpa
        })

    return pd.DataFrame(metrics)

def generate_ngrams(
    text_list: List[str],
    n: int = 1,
    freq_threshold: int = 1
) -> pd.DataFrame:
    """
    Generate n-grams from a list of preprocessed text strings and return a frequency table.
    :param text_list: List of preprocessed text strings.
    :param n: The n-gram size (1=unigrams, 2=bigrams, etc.).
    :param freq_threshold: Minimum frequency to include in the results.
    :return: DataFrame with columns: [ngram, frequency].
    """
    freq_dict = {}
    for text in text_list:
        tokens = text.split()
        for tup in ngrams(tokens, n):
            ngram_str = " ".join(tup)
            freq_dict[ngram_str] = freq_dict.get(ngram_str, 0) + 1

    # Convert to DataFrame
    data = [(ngram, freq) for ngram, freq in freq_dict.items() if freq >= freq_threshold]
    df_ngrams = pd.DataFrame(data, columns=['ngram', 'frequency'])

    # Sort by frequency descending
    df_ngrams.sort_values(by='frequency', ascending=False, inplace=True)
    df_ngrams.reset_index(drop=True, inplace=True)
    return df_ngrams

def find_collocations(
    text_list: List[str],
    freq_threshold: int = 2,
    top_n: int = 20
) -> List[Tuple[str, str, float]]:
    """
    Use BigramCollocationFinder to detect collocations (common bigrams)
    based on PMI (Pointwise Mutual Information).
    :param text_list: List of preprocessed text strings.
    :param freq_threshold: Minimum frequency for a collocation to be considered.
    :param top_n: Number of collocations to return.
    :return: List of tuples [(word1, word2, pmi_score), ...].
    """
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(
        nltk.word_tokenize(" ".join(text_list))
    )
    finder.apply_freq_filter(freq_threshold)
    scored = finder.score_ngrams(bigram_measures.pmi)
    return scored[:top_n]

def compute_tfidf(
    text_list: List[str],
    top_n: int = 20,
    min_df: float = 0.0
) -> pd.DataFrame:
    """
    Compute TF-IDF for unigrams in the text list and return top N words by average TF-IDF.
    :param text_list: List of preprocessed text strings.
    :param top_n: Number of top words to return.
    :param min_df: Minimum document frequency threshold
    :return: DataFrame with columns: [term, score].
    """
    vectorizer = TfidfVectorizer(min_df=min_df)
    try:
        tfidf_matrix = vectorizer.fit_transform(text_list)
    except ValueError as e:
        # Handle empty text list or all-zero entries
        return pd.DataFrame(columns=['term', 'score'])

    # Average TF-IDF score for each term across all documents
    avg_tfidf = tfidf_matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    df_tfidf = pd.DataFrame({'term': terms, 'score': avg_tfidf})
    df_tfidf.sort_values(by='score', ascending=False, inplace=True)
    df_tfidf.reset_index(drop=True, inplace=True)

    return df_tfidf.head(top_n)
