import nltk
import os

def download_nltk_data():
    """Download NLTK data if not already present"""
    try:
        # Create a directory for NLTK data
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Check if data already exists before downloading
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            print("NLTK data already exists")
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Warning: Error in NLTK setup: {str(e)}")
        # Continue even if NLTK setup fails
        pass

if __name__ == "__main__":
    download_nltk_data()