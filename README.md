# Question Answering System Using NLTK and TF-IDF

This project is a simple **Question Answering (QA) System** built using **Python**, **NLTK** for text preprocessing, and **TF-IDF** (Term Frequency-Inverse Document Frequency) for text vectorization. This system answers factual questions by finding the most relevant response from a set of documents based on similarity scoring.

## Features

- **Text Preprocessing**: Cleans and prepares text using NLTK with steps like tokenization, stop word removal, and lemmatization.
- **TF-IDF Vectorization**: Uses TF-IDF to convert documents and questions into numerical vectors, enabling similarity comparisons.
- **Cosine Similarity Matching**: Ranks and selects the best matching answer based on cosine similarity.
- **Efficient Retrieval**: Simple and interpretable approach that is ideal for small datasets or direct factual questions.

## Requirements

- Python 3.x
- Required libraries:
  - `nltk`
  - `scikit-learn`
  - `numpy`

To install the dependencies, run:
```bash
pip install nltk scikit-learn numpy
```

## Setup

1. **Download NLTK Data**: Run the following code to download required NLTK data (stop words, WordNet for lemmatization).
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Prepare Your Document Corpus**: Create a list of documents (each as a string) to serve as the knowledge base for the system.

## Usage

### Example Code

The code below demonstrates how to set up and query the QA system:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Sample document corpus
documents = [
    "The capital of France is Paris.",
    "Paris is known for the Eiffel Tower.",
    "The Louvre is also in Paris."
]

# Preprocess function using NLTK
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return " ".join(tokens)

# Preprocess the documents
processed_docs = [preprocess(doc) for doc in documents]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs)

# Function to get the best answer for a question
def answer_question(question):
    processed_question = preprocess(question)
    question_vector = vectorizer.transform([processed_question])
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]
    if best_match_score > 0.1:
        return documents[best_match_idx], best_match_score
    else:
        return "Sorry, I couldn't find a relevant answer.", best_match_score

# Ask a question
question = "What is the capital of France?"
answer, score = answer_question(question)
print(f"Answer: {answer} (Score: {score})")
```

### Explanation of Key Components

- **Document Corpus**: A list of documents or sentences representing the knowledge base.
- **Preprocess Function**: Cleans the text by removing stop words and lemmatizing words.
- **TF-IDF Vectorization**: Transforms the document corpus and question into vectors.
- **Cosine Similarity**: Measures similarity between the question and each document, finding the best match.
- **Answer Function**: Returns the document with the highest similarity to the question.

## Limitations

- **Contextual Limitations**: TF-IDF doesn’t capture complex word meanings or context.
- **Small Dataset Suitability**: The system is best suited for small datasets or direct factual questions.

## Possible Extensions

To improve accuracy, consider:
- **Synonym Handling**: Use word embeddings like Word2Vec or GloVe for better contextual understanding.
- **Transformer Models**: Consider using BERT or other transformers for advanced semantic matching.
- **Entity Recognition**: Identify key entities (e.g., names, dates) to improve relevance.

## License

This project is open-source and available for use and modification.

---

## Author

This QA system was created as a demonstration of using TF-IDF and NLTK for simple question-answering tasks.
