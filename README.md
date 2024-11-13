# Question Answering System Using NLTK and TF-IDF

This project is a simple **Question Answering (QA) System** built using **Python**, **NLTK** for text preprocessing, and **TF-IDF** (Term Frequency-Inverse Document Frequency) for text vectorization. This system answers factual questions by finding the most relevant response from a set of documents based on similarity scoring.

## Features

- **Text Preprocessing**: Cleans and prepares text using NLTK with steps like tokenization, stop word removal, and lemmatization.
- **TF-IDF Vectorization**: Uses TF-IDF to convert documents and questions into numerical vectors, enabling similarity comparisons.
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
```
python questions.py corpus
```

Then ask questions related to the text files in corpus.

### Explanation of Key Components

- **Document Corpus**: A list of documents or sentences representing the knowledge base.
- **Preprocess Function**: Cleans the text by removing stop words and lemmatizing words.
- **TF-IDF Vectorization**: Transforms the document corpus and question into vectors.
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