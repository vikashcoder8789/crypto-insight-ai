# Project: Crypto Insight 1.0
# Description: This file contains the code for extracting text from PDF files, preprocessing the text, and performing sentiment analysis and named entity recognition on the extracted text. It also includes a class for retrieving answers based on user queries using the BM25 algorithm.
# Authors: Amit Lakhera, Varsha, Pardipta, Krishnopriya, Vikas, Aditi, Laxmi 
# Date: 26-Feb-2025
# Version: 1.0
# Python Version: 3.8
# Libraries Used: pdfplumber, nltk, string, spacy, numpy, sklearn, textblob, rank_bm25
# Input: PDF files containing text
# Output: Extracted text, cleaned text, sentiment analysis, named entities, and answers to user queries
# Usage: Run the script and provide user queries about crypto. The script will extract text from the specified PDF files, preprocess the text, and perform sentiment analysis and named entity recognition. It will also retrieve answers to user queries using the BM25 algorithm.
# Note: The PDF files should be placed in the 'crypto_insight' folder in the same directory as this script.
# The extracted text, cleaned text, and answers will be displayed in the console.
# The sentiment analysis and named entities will be printed for each user query.    
# The script will continue to prompt for user queries until 'STOP' is entered.


# Import necessary libraries
import pdfplumber
import nltk
import string
import spacy
import numpy as np 
import re
import unidecode
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from rank_bm25 import BM25Okapi
from spacy.pipeline import EntityRuler
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure the extracted text is not None
                text += page_text + "\n"
    return text

# Get the current working directory
current_directory = os.getcwd()

# Get the relative path (which is just '.' for the current directory)
relative_path = os.path.relpath(current_directory, os.getcwd())

# List of PDF file paths
pdf_files = ['Crypto_Article1.pdf','Crypto_Article2.pdf','Crypto_Article3.pdf', 'Crypto_Article4.pdf','Crypto_Article5.pdf']  

# Extract text from multiple PDFs
all_text = ""
all_sentences = []

# Iterate through each PDF file and extract text
for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
    file_path = os.path.join(relative_path, 'crypto_insight', pdf_path)
    if os.path.exists(file_path):
        print(f"Extracting text from {file_path}...")
    else:
        print(f"File {file_path} does not exist.")
        continue  # Skip this file if it doesn't exist
    # Extract text from the PDF file
    all_text += extract_text_from_pdf(file_path) + "\n"
    
# Step 2: Preprocess the extracted text
if all_text:
    sentences = sent_tokenize(all_text)  # Sentence tokenization
    all_sentences.extend(sentences)

# Save the extracted text to a file
with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write("\n".join(all_sentences))  # Join sentences with new lines

# preprocess_text function to clean and tokenize the text
def PreProcessText(text):
    # Cleans and tokenizes input text while keeping crypto terms unchanged."""
    
    # Load stopwords
    stop_words = set(stopwords.words("english"))

    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()

    # Initialize stemmer
    stemmer = PorterStemmer()

    # Precompile regex for efficiency
    CLEAN_REGEX = re.compile(r"[^a-zA-Z0-9\s#-]")  # Allows # and -

    # Define crypto terms as a set for O(1) lookups
    crypto_terms = {"bitcoin", "ethereum", "blockchain", "dogecoin", "#ethereum", "#bitcoin", "litecoin", "ripple", "cardano", "solana", 
                    "polkadot", "chainlink", "uniswap", "binance", "coinbase", "ftx", "kraken", "defi", "nft", "metaverse", "web3", "usdt"
                    } 

    # Convert text to lowercase and normalize Unicode characters
    text = unidecode.unidecode(text.lower())
    
    # Remove unwanted characters
    text = CLEAN_REGEX.sub("", text)

    # Process words with list comprehension (efficient approach)
    return [
        word if word in crypto_terms else lemmatizer.lemmatize(word)
        for word in text.split()
            if word in crypto_terms or word not in stop_words
    ]

# Read the extracted text from the file
with open("output.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
cleaned_tokens = PreProcessText(raw_text)

# Save the cleaned text as a new file
with open("cleaned_output.txt", "w", encoding="utf-8") as file:
    file.write(" ".join(cleaned_tokens))

# Read the cleaned text from the file
with open("cleaned_output.txt", "r", encoding="utf-8") as file:
    cleaned_text = file.read()

# Step 3: Process and push the cleaned text
def ProcessAndPush(text: str):
    all_sentences = sent_tokenize(text)  # Tokenize into sentences
    # Remove duplicates and empty sentences
    #sentences = list(set(all_sentences))
    #sentences = [sent for sent in sentences if sent.strip()]
    
    vectorizer = TfidfVectorizer(max_features=5000)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    data_store = {sent: vec for sent, vec in zip(sentences, embeddings)}
    return data_store

# Step 4: Function to analyze sentiment using TextBlob
def AnalyzeCryptoSentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0:
        return "Positive", score
    elif score < 0:
        return "Negative", score
    else:
        return "Neutral", score


# Load the pre-trained SpaCy model
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Define a custom entity list for crypto-related terms
custom_crypto_entities = [
    {"label": "CRYPTOCURRENCY", "pattern": "Bitcoin"},
    {"label": "CRYPTOCURRENCY", "pattern": "Ethereum"},
    {"label": "CRYPTOCURRENCY", "pattern": "Solana"},
    {"label": "CRYPTOCURRENCY", "pattern": "Dogecoin"},
    {"label": "CRYPTOCURRENCY", "pattern": "Cardano"},
    {"label": "EXCHANGE", "pattern": "Binance"},
    {"label": "EXCHANGE", "pattern": "Coinbase"},
    {"label": "EXCHANGE", "pattern": "Kraken"},
    {"label": "EXCHANGE", "pattern": "FTX"},
    {"label": "DEFI_PLATFORM", "pattern": "Uniswap"},
    {"label": "DEFI_PLATFORM", "pattern": "Aave"},
    {"label": "DEFI_PLATFORM", "pattern": "Compound"},
    {"label": "NFT_PLATFORM", "pattern": "OpenSea"},
    {"label": "NFT_PLATFORM", "pattern": "Rarible"},
    {"label": "TOKEN", "pattern": "USDT"},
    {"label": "TOKEN", "pattern": "USDC"},
    {"label": "TOKEN", "pattern": "DAI"},
]
ruler.add_patterns(custom_crypto_entities)

# Named Entity Recognition
def ExtractCryptoEntities(text):
    # Extract crypto-related entities from text
    doc = nlp(text)
    
    # Include both default and custom entities
    crypto_entity_types = [
        "ORG", "MONEY", "PRODUCT", "GPE", "LAW", "EVENT", 
        "CARDINAL", "PERCENT", "DATE", "TIME",
        "CRYPTOCURRENCY", "EXCHANGE", "DEFI_PLATFORM", "NFT_PLATFORM", "TOKEN"
    ]
    entities = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in crypto_entity_types}
    return entities

# Step 5: Answer Retrieval using BM25
class CryptoAnswerRetrieval:
    def __init__(self, sentences):
        self.sentences = sentences
        self.tokenized_sentences = [PreProcessText(sent) for sent in sentences]
        self.bm25 = BM25Okapi(self.tokenized_sentences)
    
    #  Search function to retrieve answers based on user query
    def search(self, query, sentiment, top_n=3):
        query_tokens = PreProcessText(query.lower())
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Adjust BM25 scores based on sentiment
        if sentiment == "Negative":
            bm25_scores = [score * 0.8 for score in bm25_scores]  # Reduce score by 20%
        elif sentiment == "Positive":
            bm25_scores = [score * 1.2 for score in bm25_scores]  # Boost score by 20%

        ranked_answers = [(sent, score) for sent, score in zip(self.sentences, bm25_scores) if score > 0.1]
        ranked_answers.sort(key=lambda x: x[1], reverse=True)
        return ranked_answers[:top_n]

# Process user query
def ProcessUserQuery(user_query):
    cleaned_data = " ".join(cleaned_text.split())
    data_store = ProcessAndPush(cleaned_data)
    sentiment, score = AnalyzeCryptoSentiment(user_query)
    retriever = CryptoAnswerRetrieval(list(data_store.keys()))
    ranked_answers = retriever.search(user_query, sentiment, top_n=3)
    print(f"\nUser Query: {user_query}\n")
    print(f"Sentiment: {sentiment} (Score: {score:.2f})\n")
    print("Top-ranked Answers:")
    for i, (answer, bm25_score) in enumerate(ranked_answers):
        print(f"{i+1}. {answer[:500]}... (BM25 Score: {bm25_score:.2f})")
    return ranked_answers

# Main loop to process user queries
COINS = ["dogecoin", "bitcoin", "ethereum", "solana", "cardano"]
print("Welcome to the Crypto News Search!")
print(f"Available coins: {', '.join(COINS)}")

while True:
    coin = input("\nEnter a coin name or type 'stop' to exit: ").strip().lower()
    if coin == "stop":
        break
    if coin not in COINS:
        print("Invalid coin. Try again.")
        continue
    user_query = input("\nEnter your crypto-related query: ")
    ProcessUserQuery(user_query)