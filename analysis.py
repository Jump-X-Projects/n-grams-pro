import pandas as pd
import nltk
from nltk import ngrams
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Tuple

# If you're using collocations, NLTK sometimes needs these:
# nltk.download('punkt')

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
    # Return top_n bigrams sorted by PMI
    return scored[:top_n]

def compute_tfidf(
    text_list: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compute TF-IDF for unigrams in the text list and return top N words by average TF-IDF.
    :param text_list: List of preprocessed text strings.
    :param top_n: Number of top words to return.
    :return: DataFrame with columns: [term, tfidf_score].
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_list)

    # Average TF-IDF score for each term across all documents
    avg_tfidf = tfidf_matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    df_tfidf = pd.DataFrame({'term': terms, 'score': avg_tfidf})
    df_tfidf.sort_values(by='score', ascending=False, inplace=True)
    df_tfidf.reset_index(drop=True, inplace=True)

    return df_tfidf.head(top_n)
