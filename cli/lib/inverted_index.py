import os
from lib.search_utils import PROJECT_ROOT, load_movies, tokenize
import pickle
from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
        self.docmap_path = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

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
