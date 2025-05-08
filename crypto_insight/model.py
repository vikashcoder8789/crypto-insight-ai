# Project Name: Crypto Insight 1.1
# Project Description: This project aims to provide insights into cryptocurrency-related content by extracting and analyzing text from PDF files. It utilizes various NLP techniques, including TF-IDF and BM25 algorithms, for information retrieval and sentiment analysis.
# Authors: Amit Lakhera, Varsha, Pardipta, Krishnopriya, Vikas, Aditi, Laxmi 
# Last Modified Date: 31-March-2025
# Version: 1.1
# Python Version: 3.8
# Libraries used: pdfplumber, nltk, string, spacy, numpy, sklearn, textblob, rank_bm25, tqdm, unidecode
# Description: This script extracts text from PDF files, preprocesses the text, performs sentiment analysis and named entity recognition, and retrieves answers to user queries using the BM25 algorithm. It is designed to provide insights into cryptocurrency-related content.
# This script is designed to be run in a Python environment with the required libraries installed. It extracts text from PDF files located in the 'crypto_insight' folder, processes the text to identify cryptocurrency-related entities, and allows users to query the extracted information. The script uses various NLP techniques, including TF-IDF and BM25 algorithms, for information retrieval and sentiment analysis.
# Make sure to install them using pip if you haven't already:

import os
import re
import pdfplumber
import unidecode
import nltk
import spacy
import numpy as np
import json 
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from rank_bm25 import BM25Okapi
from spacy.matcher import PhraseMatcher
from keywords import crypto_terms


# Download necessary NLTK & spaCy data files
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

#   Set up NLTK stopwords
stopwords.words('english')


# All the coins we are using
COINS = ["bitcoin", "ethereum", "solana", "dogecoin", "hamstercoin", "cardano" ,"general crypto"]



# Create a PhraseMatcher to detect crypto-related names
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(term) for term in crypto_terms]
matcher.add("CRYPTO", patterns)

# Get current directory & locate PDFs
current_directory = os.getcwd()
pdf_folder = os.path.join(current_directory, 'crypto_insight')
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    return "\n".join(extracted_text)

# Extract text from all PDFs
all_text = ""
for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
    if os.path.exists(pdf_path):
        all_text += extract_text_from_pdf(pdf_path) + "\n\n"
    else:
        print(f"File {pdf_path} does not exist.")

# Save extracted text
with open("extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(all_text)

# Named Entity Recognition (NER) for Crypto Terms
def extract_crypto_entities(text):
    doc = nlp(text)

    # Get recognized named entities
    recognized_entities = {ent.text for ent in doc.ents if ent.label_ in ["ORG", "MONEY", "PRODUCT", "GPE"]}

    # Use PhraseMatcher for custom entity detection
    matches = matcher(doc)
    matched_crypto = {doc[start:end].text for match_id, start, end in matches}

    # Merge both sets
    crypto_entities = recognized_entities.union(matched_crypto)
    return list(crypto_entities)

crypto_entities_found = extract_crypto_entities(all_text)
#print("\nExtracted Crypto-related Entities:", crypto_entities_found)

# Text Preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = unidecode.unidecode(text.lower())
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return [word if word in crypto_terms else lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

# Sentence tokenization
sentences = sent_tokenize(all_text)

# Initialize TF-IDF model
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

# Initialize BM25 model
bm25 = BM25Okapi([preprocess_text(sent) for sent in sentences])

# Sentiment Analysis
def analyze_crypto_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    return ("Positive" if score > 0 else "Negative" if score < 0 else "Neutral"), score

# Find relevant sentences using TF-IDF
def find_tfidf_matches(query):
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    ranked_results = sorted(zip(sentences, similarity_scores), key=lambda x: x[1], reverse=True)
    return [(res[0], res[1]) for res in ranked_results if res[1] > 0.1][:3]

# Find relevant sentences using BM25
def find_bm25_matches(query, sentiment):
    query_tokens = preprocess_text(query.lower())
    bm25_scores = bm25.get_scores(query_tokens)

    # Adjust scores based on sentiment
    if sentiment == "Negative":
        bm25_scores = [score * 0.8 for score in bm25_scores]
    elif sentiment == "Positive":
        bm25_scores = [score * 1.2 for score in bm25_scores]
    
    ranked_results = [(sent, score) for sent, score in zip(sentences, bm25_scores) if score > 0.1]
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return ranked_results[:3]

# Process user query
def valid_query(query):
    query = query.lower()
    return any(keyword in query for keyword in crypto_terms)

def process_query(coin, query):
    modified_query = f"{coin} {query}"

    sentiment, score = analyze_crypto_sentiment(modified_query)

    bm25_results = find_bm25_matches(modified_query, sentiment)  # [(sentence, score), ...]
    tfidf_results = find_tfidf_matches(modified_query)  # [(sentence, score), ...]

    # Merge results and keep the highest score for each sentence
    combined_results = {}
    for sentence, score in bm25_results + tfidf_results:
        if sentence in combined_results:
            combined_results[sentence] = max(combined_results[sentence], score)
        else:
            combined_results[sentence] = score

    # Sort results based on score (highest first)
    ranked_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)

    # Limit to top 3 answers
    top_results = ranked_results[:3]


    return {
    "query": query,
    "coin": coin,
    "sentiment": f"{sentiment} (Score: {score:.2f})",
    "top_answers": [
        f"{i+1}. {answer[:600]}..."
        for i, (answer, score) in enumerate(top_results)
    ],
    "raw_top_results": top_results  # This line is for logging
}


=======
    print(f"\nUser Query: {user_query}")
    print(f"Sentiment: {sentiment} (Score: {score:.2f})\n")
    print("Top-ranked Answers:")
    for i, (answer, score) in enumerate(top_results):
        print(f"{i+1}. {answer[:500]}... (Score: {score:.2f})")
    return top_results, sentiment, score

# Log user queries and results
LOG_FILE = os.path.join(current_directory, "query_logs.json")
def log_query(coin, user_query, sentiment, top_results):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "coin": coin,
        "query": user_query,
        "sentiment": sentiment,
        "answers": [
            {
                "text": sentence,
                "score": round(score, 4)
            } for sentence, score in top_results
        ]
    }
        # Append to JSON file
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        else:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=4)
    except Exception as e:
        print(f"Error logging query: {e}")

# Main loop to process user queries
COINS = ["dogecoin", "bitcoin", "ethereum", "solana", "cardano"]
print("\nWelcome to the Crypto Insight 1.1")
print(f"\nAvailable coins: {', '.join(COINS)}")

while True:
    coin = input("\nEnter a coin name or type 'stop' to exit: ").strip().lower()
    if coin == "stop":
        break

    if coin not in COINS:
        print("\nInvalid coin. Try again.")
        continue

    user_query = input("\nEnter your crypto-related query: ")
    top_results, sentiment, score = process_user_query(user_query)
    log_query(coin, user_query, sentiment, top_results)
    print("\nQuery logged successfully.")

