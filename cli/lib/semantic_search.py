import os
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np

from lib.search_utils import PROJECT_ROOT, load_movies


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = defaultdict()
        self.embeddings_path = os.path.join(PROJECT_ROOT, "cache", "movie_embeddings.npy")

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = np.load(f)

        if self.embeddings is not None and len(self.embeddings) == len(self.documents):
            return self.embeddings

        return self.build_embeddings(documents)



    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}:{doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def generate_embedding(self, text: str):
        if text == "" or text.isspace():
            raise ValueError("No input text provided")

        embedding = self.model.encode([text])
        return embedding[0]


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()

    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(f"Embedding shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")




def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
