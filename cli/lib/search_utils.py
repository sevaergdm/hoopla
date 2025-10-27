import json
import os
import string

from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
BM25_K1 = 1.5
BM25_B = 0.75


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data.get("movies")


def load_stopwords() -> list:
    with open(STOPWORDS_PATH, "r") as f:
        lines = f.read().splitlines()
    return lines

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize(input: str) -> list:
    text = preprocess_text(input)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stopwords = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stopwords:
            filtered_words.append(word)

    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def remove_stopwords(input: list, stopwords: list) -> list:
    output = []
    for word in input:
        if word in stopwords:
            continue
        output.append(word)
    return output
