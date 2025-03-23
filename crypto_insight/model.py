#Step 1: Done By Varsha 
import pdfplumber
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('punkt_tab')
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure the extracted text is not None
                text += page_text + "\n"
    return text

# List of PDF file paths
pdf_files = ['Crypto_Article1.pdf','Crypto_Article2.pdf','Crypto_Article3.pdf','Crypto_Article4.pdf','Crypto_Article5.pdf']  # Replace with your actual file paths

# Extract text from multiple PDFs
all_text = ""
for pdf_path in pdf_files:
    all_text += extract_text_from_pdf(pdf_path) + "\n"

# Print extracted text
print(all_text)

# Save the extracted text to a file
with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write(all_text)

#Step 2: Pre-processisng  Done By Varsha

# Load English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]

    return filtered_words

# Read the extracted text
with open("output.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Apply preprocessing
cleaned_tokens = preprocess_text(raw_text)

# Print sample cleaned tokens
print(cleaned_tokens[:50])  # Print first 50 words after preprocessing

# Save the cleaned text as a new file
with open("cleaned_output.txt", "w", encoding="utf-8") as file:
    file.write(" ".join(cleaned_tokens))

# Step 3 : Done By Vikas vectorization Split

def processAndPush(text: str):
    sentences = sent_tokenize(text)  # Sentence splitting

    # Using TF-IDF embeddings as nltk doesn't have built-in word vectors
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # Simulating "pushing" data (storing embeddings)
    data_store = {sent: vec for sent, vec in zip(sentences, embeddings)}

    return data_store  # This is where you'd actually push to a database

# Example usage

result = processAndPush(text)
print(result)  # Prints sentence-to-vector mapping
