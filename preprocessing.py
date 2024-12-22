import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from typing import Union, List, Optional, Dict

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def load_csv_in_chunks(
    file,
    text_column: str,
    chunksize: int = 100000,
    max_rows: int = 1000000
) -> pd.DataFrame:
    """
    Efficiently load a CSV file in chunks (up to 1 million rows).
    :param file: The uploaded file handle from Streamlit.
    :param text_column: Name of the column containing text data.
    :param chunksize: Number of rows to load per chunk.
    :param max_rows: Maximum number of rows to load overall.
    :return: A pandas DataFrame with text.
    """
    try:
        # Read the first chunk to get column names
        first_chunk = pd.read_csv(file, nrows=1)

        # Check if the column exists (case-sensitive)
        if text_column not in first_chunk.columns:
            # Try to find a case-insensitive match
            col_map = {col.lower(): col for col in first_chunk.columns}
            if text_column.lower() in col_map:
                text_column = col_map[text_column.lower()]
            else:
                available_columns = "\n- ".join(first_chunk.columns)
                raise ValueError(f"Column '{text_column}' not found. Available columns are:\n- {available_columns}")

        # Reset file pointer
        file.seek(0)

        # Read the file in chunks
        dfs = []
        rows_loaded = 0
        for chunk in pd.read_csv(file, chunksize=chunksize):
            dfs.append(chunk[[text_column]])
            rows_loaded += len(chunk)
            if rows_loaded >= max_rows:
                break

        if not dfs:
            raise ValueError("No data was read from the file")

        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {str(e)}")

def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    remove_punctuation: bool = True,
    lowercase: bool = True,
    custom_stopwords: Optional[List[str]] = None
) -> str:
    """
    Apply basic preprocessing to a single text string.
    :param text: The original text.
    :param remove_stopwords: Flag to remove stopwords.
    :param remove_punctuation: Flag to remove punctuation.
    :param lowercase: Flag to convert text to lowercase.
    :param custom_stopwords: Additional user-provided stopwords.
    :return: Cleaned text string.
    """
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # Remove punctuation using regex
        text = re.sub(r'[^\w\s]', '', text)

    tokens = text.split()

    # Merge custom stopwords into our global set if provided
    combined_stopwords = STOPWORDS.copy()
    if custom_stopwords:
        combined_stopwords.update([w.lower() for w in custom_stopwords])

    if remove_stopwords:
        tokens = [t for t in tokens if t not in combined_stopwords]

    return " ".join(tokens)

def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str,
    remove_stopwords: bool = True,
    remove_punctuation: bool = True,
    lowercase: bool = True,
    custom_stopwords: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply text preprocessing to an entire dataframe's text column.
    :param df: The DataFrame containing a column of text.
    :param text_column: The column name to preprocess.
    :param remove_stopwords: Flag to remove stopwords.
    :param remove_punctuation: Flag to remove punctuation.
    :param lowercase: Flag to convert text to lowercase.
    :param custom_stopwords: Additional user-provided stopwords.
    :return: DataFrame with a new column 'processed_text'.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Handle missing values
    df[text_column] = df[text_column].fillna('')

    # Convert to string type using the new pandas string accessor
    df[text_column] = df[text_column].astype('string')

    df['processed_text'] = df[text_column].apply(
        lambda x: preprocess_text(
            str(x),
            remove_stopwords=remove_stopwords,
            remove_punctuation=remove_punctuation,
            lowercase=lowercase,
            custom_stopwords=custom_stopwords
        )
    )
    return df
