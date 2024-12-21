# Jump-n-Grams Streamlit App

A simple Streamlit application for exploring **jump-n-grams** (also known as skip-grams) in text data. This tool lets you upload text, configure parameters (e.g., n-gram size, jump size), and view the most frequent jump-n-grams in both tabular and chart form.

---

## Features

- **Upload or Use Sample Data**
  Option to upload your own `.txt`, `.csv`, or `.md` file, or simply analyze a built-in sample.

- **Preprocessing**
  Choose to lowercase text and/or remove punctuation before generating jump-n-grams.

- **Dynamic Parameter Control**
  - **N-gram size (n)**: how many tokens make up one n-gram (e.g., 2 for bigrams, 3 for trigrams).
  - **Jump**: how many tokens to skip between each token in an n-gram. (jump=0 implies standard n-grams.)

- **Fast Computation & Caching**
  Under the hood, a `@st.cache_data` decorator ensures repeated computations on the same text aren’t re-run.

- **Interactive Visualizations**
  Bar charts showing the most frequent jump-n-grams, along with a data table you can browse.

- **Download Results**
  Export your jump-n-gram frequency table as a CSV file for further analysis.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
   cd <REPO_NAME>
