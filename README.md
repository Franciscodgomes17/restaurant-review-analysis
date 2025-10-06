# Restaurant Review Analysis: Multi-Label Classification & Sentiment Analysis

A comprehensive NLP project analyzing Hyderabad restaurant reviews using advanced text mining techniques including multi-label classification, sentiment analysis, topic modeling, and co-occurrence network analysis.

## Project Overview

This project tackles the "Hyderabadi Word Soup" by analyzing restaurant reviews to:
- **Classify cuisine types** from review content using multiple ML approaches
- **Analyze sentiment** using VADER and TextBlob, comparing with actual ratings
- **Discover latent topics** in reviews using LSA and LDA
- **Visualize dish co-occurrence patterns** using network analysis

## Key Features

### 🔍 Multi-Label Classification
- **Vectorization Methods**: TF-IDF, Word2Vec (Skip-gram & CBOW), GloVe
- **Classification Strategies**: OneVsRest, ClassifierChain
- **Models Tested**: Logistic Regression, Decision Trees, Random Forest, MLP, Dummy Classifier
- **Best Model**: MLP Classifier with TF-IDF + ClassifierChain

### Sentiment Analysis
- **VADER Implementation**: Compound scores and sentence-level analysis
- **TextBlob Comparison**: Polarity and subjectivity analysis
- **Performance**: VADER showed better correlation with ratings (0.70 vs 0.69)

### Topic Modeling
- **Techniques**: LSA and LDA with BoW and TF-IDF
- **Coherence Scores**: LSA with BoW achieved highest coherence (0.4536)
- **Topics Discovered**: Food quality, service experience, delivery performance

### Co-occurrence Analysis
- **Network Analysis**: Dish pairing patterns across cuisines
- **Community Detection**: 5 distinct dish clusters identified
- **Key Insights**: Chicken-based dishes form central nodes in co-occurrence networks

## Project Structure
Text_Mining_Restaurant_Analysis/
├── 📊 Notebook_Group_8.ipynb # Main analysis notebook
├── 🔧 Functions_Group_8.py # Utility functions module
├── 📁 data/ # Dataset directory
│ ├── 10k_reviews.csv
│ ├── 105_restaurants.csv
│ └── glove/ # GloVe embeddings
├── 📄 requirements.txt # Dependencies
├── 📄 README.md # Project documentation
└── 📄 LICENSE # MIT License


## Installation & Setup

```bash
# Clone repository
git clone https://github.com/your-username/restaurant-review-analysis.git
cd restaurant-review-analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
