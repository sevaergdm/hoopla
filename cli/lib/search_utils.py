import json
import os
import string
from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data.get("movies")


def load_stopwords() -> list:
    with open(STOPWORDS_PATH, "r") as f:
        lines = f.read().splitlines()
    return lines


def format_text(input: str) -> str:
    output = input.lower()
    output = output.translate(str.maketrans("", "", string.punctuation))
    return output


def tokenize(input: str) -> list:
    stemmer = PorterStemmer()
    stopwords = load_stopwords()
    output = []
    for word in input.split(" "):
        word = format_text(word)
        if word and word not in stopwords:
            stem = stemmer.stem(word)
            output.append(stem)

    return output

def remove_stopwords(input: list, stopwords: list) -> list:
    output = []
    for word in input:
        if word in stopwords:
            continue
        output.append(word)
    return output
