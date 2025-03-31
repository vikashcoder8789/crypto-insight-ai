import pdfplumber
import nltk
import string
import numpy as np
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from multiprocessing import Pool

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

### Step 1: Data Collection (Extract Text from PDF in Chunks)
def extractTextFromPDF(pdf_path, page_limit=100):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= page_limit:  # Process only `page_limit` pages at a time
                break
            text += page.extract_text() + "\n"
    return text

### Step 2: Pre-processing (Stopword Removal, Stemming, Lemmatization)
def preprocessText(text):
    words = word_tokenize(text.lower())  # Tokenization & lowercase
    words = [word for word in words if word not in string.punctuation]  # Remove punctuation
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    processed_words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in filtered_words]  # Stemming & Lemmatization
    return ' '.join(processed_words)  # Return cleaned text

### Step 3: Splitting Feed NLP (Sentence Embeddings)
def processAndPush(text):
    sentences = sent_tokenize(text)  # Sentence Splitting
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
    embeddings = vectorizer.fit_transform(sentences).toarray()  # Convert sentences to TF-IDF vectors
    
    data_store = {sent: vec for sent, vec in zip(sentences, embeddings)}  # Store sentence embeddings
    return data_store

### Step 4: Sentiment Analysis (Parallel Processing)
def analyze_sentiment(sentence):
    blob = TextBlob(sentence)
    return sentence, blob.sentiment.polarity

def sentimentAnalysisParallel(sentences):
    with Pool() as pool:
        results = pool.map(analyze_sentiment, sentences)
    return dict(results)

### Step 5: Answer Ranking (BM25)
def rankAnswers(sentences, query):
    tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in sentences]
    tokenized_query = word_tokenize(query.lower())
    
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    
    ranked_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return ranked_sentences[:3]  # Return top 3 ranked answers

### Example Usage:
pdf_path = "D:\file\crypto\crypto news data.pdf"  # Path to your PDF file
raw_text = extractTextFromPDF(pdf_path, page_limit=100)  # Step 1: Extract text (100 pages at a time)
clean_text = preprocessText(raw_text)  # Step 2: Preprocess text
data_store = processAndPush(clean_text)  # Step 3: Process and store data
sentiment_scores = sentimentAnalysisParallel(list(data_store.keys()))  # Step 4: Sentiment Analysis
ranked_answers = rankAnswers(list(data_store.keys()), "Bitcoin price")  # Step 5: Rank Answers

# Display results
print("\nSentiment Scores:")
for sentence, score in sentiment_scores.items():
    print(f"Sentiment: {score:.4f} - {sentence}")

print("\nTop Ranked Answers:")
for sentence, score in ranked_answers:
    print(f"Score: {score:.4f} - {sentence}")
