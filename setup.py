import nltk

def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

if __name__ == "__main__":
    download_nltk_data()