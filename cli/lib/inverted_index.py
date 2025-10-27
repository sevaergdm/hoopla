import math
import os
import pickle
from collections import Counter, defaultdict
from itertools import islice
from typing import OrderedDict

from lib.search_utils import (BM25_B, BM25_K1, PROJECT_ROOT, load_movies,
                              tokenize)


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
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(PROJECT_ROOT, "cache", "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        total_tokens = len(tokens)
        self.doc_lengths[doc_id] = total_tokens
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        total_length = 0
        total_docs = len(self.doc_lengths)
        for length in self.doc_lengths.values():
            total_length += length
        
        return total_length / total_docs

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        return self.term_frequencies[doc_id][token]

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        raw_tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        return idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)


    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    
    def bm25_search(self, query: str, limit: int) -> dict[int, float]:
        tokens = tokenize(query)
        scores = defaultdict(float)

        for token in tokens:
            docs = self.get_documents(token)
            for doc in docs:
                scores[doc] += self.bm25(doc, token)

        sorted_scores = OrderedDict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
        return dict(islice(sorted_scores.items(), limit))


    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie.get("id", 0)
            text = f"{movie.get('title', '')} {movie.get('description')}"
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

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

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

        try:
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except Exception as e:
            print(f"Unable to open doc lengths file: {e}")
