import json
import os
import re
from collections import defaultdict
from itertools import islice
from typing import OrderedDict

import numpy as np
from lib.search_utils import (DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
                              DEFAULT_MAX_CHUNK_SIZE, PROJECT_ROOT,
                              format_search_result, load_movies)
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = defaultdict()
        self.embeddings_path = os.path.join(
            PROJECT_ROOT, "cache", "movie_embeddings.npy"
        )

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

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded")

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded")

        embedding = self.generate_embedding(query)

        scores = []
        for i, e in enumerate(self.embeddings):
            similarity = cosine_similarity(embedding, e)
            scores.append((similarity, self.documents[i]))

        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)

        result = []
        for score, doc in sorted_scores[:limit]:
            result.append(
                {
                    "doc_id": doc.get("id"),
                    "score": score,
                    "title": doc.get("title"),
                    "description": doc.get("description"),
                }
            )

        return result


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(
            PROJECT_ROOT, "cache", "chunk_embeddings.npy"
        )
        self.chunk_metadata_path = os.path.join(
            PROJECT_ROOT, "cache", "chunk_metadata.json"
        )

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        metadata = []

        for i, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunking(
                text, DEFAULT_MAX_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
            )
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append(
                    {
                        "movie_idx": i,
                        "chunk_idx": j,
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)

            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)
        chunk_score = []

        if self.chunk_embeddings is None:
            return []

        if self.chunk_metadata is None:
            return []

        for i, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk)
            chunk_metadata = self.chunk_metadata[i]
            chunk_score.append(
                {
                    "chunk_idx": i,
                    "movie_idx": chunk_metadata["movie_idx"],
                    "score": score,
                }
            )

        movies_to_scores = defaultdict()
        for score in chunk_score:
            if (
                movies_to_scores.get(score["movie_idx"]) is None
                or movies_to_scores[score["movie_idx"]] < score["score"]
            ):
                movies_to_scores[score["movie_idx"]] = score["score"]

        movies_to_scores = OrderedDict(
            sorted(movies_to_scores.items(), key=lambda kv: kv[1], reverse=True)
        )
        top_movies = dict(islice(movies_to_scores.items(), limit))

        results = []
        for k, v in top_movies.items():
            result = format_search_result(
                str(k),
                self.documents[k]["title"],
                self.documents[k]["description"][:100],
                v,
            )
            results.append(result)

        return results


def search_chunked(query: str, limit: int = 10) -> list[dict]:
    movies = load_movies()
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(movies)

    return search.search_chunks(query, limit)


def embed_chunks():
    movies = load_movies()
    search = ChunkedSemanticSearch()
    return search.load_or_create_chunk_embeddings(movies)


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunking(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semantic_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    pattern = r"(?<=[.!?])\s+"
    text = text.strip()
    if text == "":
        return []

    split_text = re.split(pattern, text)
    if len(split_text) == 1 and not text.endswith((".", "?", "!")):
        split_text = [text]

    chunks = []
    i = 0
    n_sentences = len(split_text)
    while i < n_sentences - overlap:
        chunk = split_text[i : i + max_chunk_size]
        cleaned_sentences = []
        for chunk_sentence in chunk:
            cleaned_sentences.append(chunk_sentence.strip()) 
        if not cleaned_sentences:
            continue

        chunks.append(" ".join(cleaned_sentences))
        i += max_chunk_size - overlap
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")

    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        if i < overlap:
            chunk_words = words[i : i + chunk_size]
        else:
            chunk_words = words[i - overlap : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size

    return chunks


def search_command(query: str, limit: int):
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)

    results = semantic_search.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()

    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(
        f"Embedding shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


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
