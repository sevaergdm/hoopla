import math
import os
import pickle
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from lib.search_utils import PROJECT_ROOT, load_movies, tokenize


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
        self.docmap_path = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.term_frequencies_path = os.path.join(
            PROJECT_ROOT, "cache", "term_frequencies.pkl"
        )

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        print(PorterStemmer().stem("trapper"))
        print(PorterStemmer().stem("strapper"))
        doc_count = len(self.docmap)
        print(doc_count)
        term_doc_count = len(self.index[token])
        print(term_doc_count)
        print(self.index[token])
        print(self.term_frequencies[424][token])
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        return idf

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        print(f"TF: {tf}")
        idf = self.get_idf(term)
        print(f"IDF: {idf}")
        return tf * idf

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie.get("id", 0)
            text = f"{movie.get('title', '')}{movie.get('description')}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(os.path.join(PROJECT_ROOT, "cache"), exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
        except Exception as e:
            print(f"Unable to open index file: {e}")

        try:
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except Exception as e:
            print(f"Unable to open docmap file: {e}")

        try:
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except Exception as e:
            print(f"Unable to open term frequencies file: {e}")
