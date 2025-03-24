#Step 1: Done By Varsha 
import pdfplumber
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from rank_bm25 import BM25Okapi

#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure the extracted text is not None
                text += page_text + "\n"
    return text

# Added by Amit for Automatic path detection (Current Directory)
import os
# Get the current working directory
current_directory = os.getcwd()
# Get the relative path (which is just '.' for the current directory)
relative_path = os.path.relpath(current_directory, os.getcwd())

# List of PDF file paths
pdf_files = ['Crypto_Article1.pdf','Crypto_Article2.pdf','Crypto_Article3.pdf','Crypto_Article4.pdf','Crypto_Article5.pdf']  
# Replace with your actual file paths

# Extract text from multiple PDFs
all_text = ""
all_sentences = []

for pdf_path in pdf_files:
    file_path = "".join([relative_path, '\\crypto_insight\\' ,pdf_path])
    all_text += extract_text_from_pdf(file_path) + "\n"
    
if all_text:
        sentences = sent_tokenize(all_text)  # Sentence tokenization
        all_sentences.extend(sentences)

# Print extracted text
#print(all_text)

# Save the extracted text to a file
with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write("".join(all_sentences))  # Join sentences with new lines

#print(all_sentences);

#Step 2: Pre-processisng  Done By Varsha

# Load English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    #words = word_tokenize(text)
    #print("text:",text)

    # Remove stopwords
    filtered_words = [word for word in text.split() if word not in stop_words]
    #print("filtered_words:",filtered_words)

    return filtered_words

# Read the extracted text
with open("output.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Apply preprocessing
cleaned_tokens = preprocess_text(raw_text)

# Save the cleaned text as a new file
with open("cleaned_output.txt", "w", encoding="utf-8") as file:
    file.write(" ".join(cleaned_tokens))

#  Step 3 : Done By Vikas vectorization Split
def processAndPush(text: str):
    #sentences = sent_tokenize(text)  # Sentence splitting

    # Using TF-IDF embeddings as nltk doesn't have built-in word vectors
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # Simulating "pushing" data (storing embeddings)
    data_store = {sent: vec for sent, vec in zip(sentences, embeddings)}

    return data_store  # This is where you'd actually push to a database



# Example usage
#result = processAndPush(text)
#print(result)  # Prints sentence-to-vector mapping

#print("step 3 done")

# Step 4: Added for Sentiment Analysis using TextBlob - Amit Lakhera
# Function to analyze sentiment using TextBlob
def analyze_crypto_sentiment(text):
    # Analyze sentiment using TextBlob
    blob = TextBlob(text)
    score = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
    
    if score > 0:
        return "Positive", score
    elif score < 0:
        return "Negative", score
    else:
        return "Neutral", score

    
class CyptoAnswerRetrieval:
    def __init__(self, sentences):

        # Initialize BM25 with preprocessed sentences
        self.sentences = sentences
        self.tokenized_sentences = [preprocess_text(sent) for sent in sentences]
        self.bm25 = BM25Okapi(self.tokenized_sentences)
    
    def search(self, query, top_n=3):
        # Retrieve the most relevant answers based on BM25 score
        query_tokens = preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_answers = sorted(zip(self.sentences, scores), key=lambda x: x[1], reverse=True)
        return ranked_answers[:top_n]

# Step 5: Added for Process user query and retrieve answers - Amit Lakhera
def process_user_query(user_query):  
      
    # Step 1: Extract Text from PDFs
    cleaned_data = " ".join(cleaned_tokens)
    data_store = processAndPush(cleaned_data)
    print("Documents extracted and processed.")
    
    # Step 2: Sentiment Analysis
    sentiment, score = analyze_crypto_sentiment(user_query)
    print("Sentiment Analysis completed.")
    
    # Step 3: Search for Relevant Answer
    retriever = CyptoAnswerRetrieval(data_store)
    ranked_answers = retriever.search(user_query, top_n=3)
    print("Answer retrieval completed.")
    
    print(f"\nUser Query: {user_query}\n")
    print(f"\nUser Query Sentiment: {sentiment} (Score: {score})\n")
    print("Top-ranked Answers:")
    
    for i, (answer, bm25_score) in enumerate(ranked_answers):
        print(f"{i+1}. {answer[:500]}... (BM25 Score: {bm25_score:.2f})")
    return ranked_answers

user_query = input("Enter your query about crypto: ")
ranked_answers = process_user_query(user_query)