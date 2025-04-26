# Crypto Text Analysis Pipeline

Welcome to the "Crypto Text Analysis Pipeline", a Python-based tool designed to process cryptocurrency-related articles, extract meaningful insights, and answer user queries. This modular pipeline scrapes web articles, converts them to PDFs, preprocesses text, performs sentiment analysis, and retrieves ranked answers using the BM25 algorithm. Built with contributions from multiple authors, it leverages powerful NLP and machine learning libraries to deliver robust cryptocurrency insights.

## Objectives

- **Data Acquisition:** Scrape cryptocurrency news articles from the web (e.g., thenewscrypto.com) for five coins: Dogecoin, Bitcoin, Ethereum, Solana, and Hamster.

- **Text Extraction & Preprocessing:** Convert articles to PDFs, extract clean text, and refine it for analysis while preserving crypto-specific terms.

- **Sentiment Analysis:** Analyze the tone of articles (positive, negative, neutral) and visualize sentiment distribution.

- **Query Response:** Deliver ranked, relevant answers to user queries based on processed text using the BM25 algorithm.

## Key Features

### Data Acquisition

- Scrapes up to 10 article URLs per coin from the first page of search results on thenewscrypto.com.
- Saves URLs in CSV files (e.g., dogecoin_news_urls.csv) with robust error handling and logging.
- Uses a user-agent header to mimic browser requests and avoid blocking.

### Text Extraction

- Converts articles to PDFs using pdfkit and extracts clean text with trafilatura, removing boilerplate content (e.g., ads, navigation).
- Outputs structured text ready for analysis.

### Data Preprocessing

- Cleans text by lowercasing, removing special characters, tokenizing, stemming, and lemmatizing.
- Preserves cryptocurrency-specific terms for domain accuracy.
- Saves cleaned output to cleaned_data.txt with a progress bar for large datasets.

### Sentiment Analysis

- Analyzes sentence-level sentiment using TextBlob (polarity and subjectivity scores).
- Categorizes sentiments as positive, negative, or neutral and visualizes distribution with matplotlib.

### Answer Retrieval

- Indexes preprocessed text with the BM25Okapi algorithm for efficient query matching.
- Returns top-ranked sentences with metadata (source, score, sentiment) for user queries.

### Web Interface & API (New)

- Flask-based web interface for easy interaction with the system
- RESTful API endpoint for querying the crypto analysis engine
- Cross-origin support through Flask-CORS for frontend integration
- Interactive form for selecting cryptocurrencies and entering queries
- Persistent query logging for analytics and debugging

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Pipeline  │───►│  NLP Processing │───►│ BM25 Indexing   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                            │
         │                                            ▼
         │                                    ┌─────────────────┐
         │                                    │ Flask Backend   │
         └───────────────────────────────────►│  (app.py)      │
                                              └─────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────┐
                                              │  Web Frontend   │
                                              │  (index.html)   │
                                              └─────────────────┘
```

## Setup Instructions

### Prerequisites

- **Python Version:** 3.8+
  
- **External Dependency:** Install `wkhtmltopdf` for PDF conversion (required by `pdfkit`):
  
### Installation

1. Clone the repository:
   ```
   git clone https://github.com/techiepookie/crypto-insight-ai.git
   cd crypto-insight-ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Requirements File (requirements.txt)

```
requests
beautifulsoup4
trafilatura
pdfkit
nltk
textblob
matplotlib
rank-bm25
flask
flask-cors
```

## Usage

### Run the Pipeline:

Execute the main script to scrape, process, and analyze data:

```bash
python main.py
```

This generates CSV files (URLs), PDFs, cleaned text (cleaned_data.txt), sentiment visualizations, and a searchable BM25 index.

### Command-line Query Interface:

Use the provided query interface to ask questions:

```python
results = retrieval.search_with_bm25("What is the latest trend in Bitcoin price?", top_n=3)
print(results)
```

Output includes ranked sentences with source, score, and sentiment.

### Web Interface :

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser at http://localhost:5000

3. Select a cryptocurrency from the dropdown, enter your query, and submit.

4. View the sentiment analysis and top answers directly in your browser.

### API Usage :

Send POST requests to the `/query` endpoint:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"coin": "bitcoin", "query": "What is the latest trend in Bitcoin price?"}'
```

Response format:
```json
{
  "query": "What is the latest trend in Bitcoin price?",
  "coin": "bitcoin",
  "sentiment": "Positive (Score: 0.65)",
  "top_answers": [
    "1. Bitcoin prices surged this week due to increased institutional adoption. (Score: 0.92)",
    "2. Analysts predict Bitcoin price trends will stabilize in Q2 2025. (Score: 0.75)",
    "3. Bitcoin faced a price dip amid regulatory concerns. (Score: 0.68)"
  ]
}
```

## Libraries Used

1. **Requests:** HTTP requests for web scraping.
2. **BeautifulSoup (bs4):** HTML parsing for article URLs.
3. **Trafilatura:** Clean text extraction from articles.
4. **Pdfkit:** HTML-to-PDF conversion.
5. **NLTK:** Tokenization, stemming, and lemmatization.
6. **TextBlob:** Sentiment analysis.
7. **Matplotlib:** Visualization of sentiment distribution.
8. **Rank-BM25:** Ranked answer retrieval.
9. **Flask:** Web framework for backend API and frontend serving.
10. **Flask-CORS:** Cross-Origin Resource Sharing support.

## Backend Components 

### app.py
- Main Flask application entry point
- Defines web routes (/, /query)
- Handles JSON request/response processing
- Integrates with NLP model components

### logger.py
- Manages query logging to JSON file
- Captures query history with timestamps
- Records sentiment scores and top answers

### model.py 
- Adapted to work as a backend service
- process_query() function receives coin and query parameters
- Returns structured responses for JSON serialization

### keywords.py
- Maintains cryptocurrency-specific terms
- Supports context-aware text processing
- Prevents removal of domain terminology

## Example Output:

**Query:** "What is the latest trend in Bitcoin price?"

**Result:**
1. Source: bitcoin_article_3.pdf | Score: 0.92 | Sentiment: Positive
   "Bitcoin prices surged this week due to increased institutional adoption."

2. Source: bitcoin_article_1.pdf | Score: 0.75 | Sentiment: Neutral
   "Analysts predict Bitcoin price trends will stabilize in Q2 2025."

3. Source: bitcoin_article_5.pdf | Score: 0.68 | Sentiment: Negative
   "Bitcoin faced a price dip amid regulatory concerns."

## Contributing

We welcome contributions! Please fork the repository, create a branch, and submit a pull request with your changes. For major updates, open an issue first to discuss.

## Contact

Please submit a pull request or file an issue on GitHub for support or inquiries.
