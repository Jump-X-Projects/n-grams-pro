import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {str(e)}")

def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        print("Warning: Using empty stopwords set as NLTK data couldn't be loaded")
        return set()

def preprocess_dataframe(df, text_column, remove_stopwords=True, 
                        remove_punctuation=True, lowercase=True,
                        custom_stopwords=None):
    """
    Preprocess text data in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing text data
    text_column : str
        Name of column containing text to process
    remove_stopwords : bool
        Whether to remove stopwords
    remove_punctuation : bool
        Whether to remove punctuation
    lowercase : bool
        Whether to convert text to lowercase
    custom_stopwords : list
        Additional stopwords to remove
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed text column added
    """
    
    # Create a copy of the DataFrame
    df_processed = df.copy()
    
    # Get stopwords if needed
    if remove_stopwords:
        stop_words = get_stopwords()
        if custom_stopwords:
            stop_words.update(custom_stopwords)
    
    def process_text(text):
        # Convert to string if not already
        text = str(text)
        
        # Lowercase if requested
        if lowercase:
            text = text.lower()
            
        # Remove punctuation if requested
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Split into words
        words = text.split()
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = [w for w in words if w not in stop_words]
            
        # Handle Google Ads specific patterns
        words = clean_ads_terms(words)
            
        return ' '.join(words)
    
    # Apply preprocessing to text column
    df_processed['processed_text'] = df_processed[text_column].apply(process_text)
    
    return df_processed

def clean_ads_terms(words):
    """Clean Google Ads specific patterns from search terms."""
    cleaned_words = []
    for word in words:
        # Remove special Google Ads markers
        word = re.sub(r'\+', '', word)  # Remove broad match modifiers
        word = re.sub(r'^\[|\]$', '', word)  # Remove exact match brackets
        word = re.sub(r'^\"|\"$', '', word)  # Remove phrase match quotes
        
        # Remove common Google Ads artifacts
        word = re.sub(r'near\s?me', '', word)  # Remove "near me"
        word = re.sub(r'best', '', word)  # Remove generic quality terms
        
        if word:  # Only add non-empty words
            cleaned_words.append(word)
    
    return cleaned_words