import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def extract_bigrams(text):
    preprocessed_text = preprocess_text(text)
    # Tokenize the preprocessed text into words
    tokens = word_tokenize(preprocessed_text)
    # Generate bigrams
    bigrams = list(ngrams(tokens, 3))
    # Convert bigrams to string format
    bigram_strings = [' '.join(bigram) for bigram in bigrams]
    return bigram_strings

def perform_keyword_analysis(text):
    preprocessed_text = preprocess_text(text)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Get the TF-IDF scores for each feature
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Create a list of tuples (word, TF-IDF score) and sort it by the score
    keyword_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

    # Extract the top keywords (adjust the number based on your needs)
    top_keywords = [keyword for keyword, score in keyword_scores[:10]]

    return top_keywords

# Example text
sample_text = """
Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans using natural language. It involves the development of algorithms and models that enable computers to understand, interpret, and generate human-like text. NLP is used in various applications, including chatbots, language translation, sentiment analysis, and more.
"""

# Extract paired keywords (bigrams)
bigrams = extract_bigrams(sample_text)
print("Bigrams:", bigrams)

# Perform keyword analysis on the example text
keywords = perform_keyword_analysis(sample_text)
print("Top Keywords:", keywords)
