import os

from lib.search_utils import DEFAULT_K_VALUE, format_search_result, load_movies

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_search_result = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_result, semantic_search_result, alpha)

        return combined[:limit]

    def rrf_search(self, query: str, k, limit: int = 10):
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)

        combined = rrf_combine_search_results(bm25_result, semantic_result, k)

        return combined[:limit]


def rrf_search_command(query: str, k: int = DEFAULT_K_VALUE, limit: int = 5) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    result = hybrid_search.rrf_search(query, k, limit)
    return {
        "original_query": query,
        "query": query,
        "k": k,
        "results": result,
    }


def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    result = hybrid_search.weighted_search(query, alpha, limit)

    return {
        "original_query": query,
        "query": query,
        "alpha": alpha,
        "results": result,
    }


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    norm_scores = []
    for score in scores:
        norm_scores.append((score - min_score) / (max_score - min_score))

    return norm_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores = []
    for result in results:
        scores.append(result["score"])

    norm_scores = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = norm_scores[i]

    return results


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = 0.5
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def rrf_score(rank: int, k: int = DEFAULT_K_VALUE) -> float:
    return 1 / (rank + k)


def rrf_combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], k: int = DEFAULT_K_VALUE
):
    combined = {}

    for i, result in enumerate(bm25_results, 1):
        doc_id = result["id"]
        if doc_id not in combined:
            combined[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": 0,
                "semantic_rank": 0,
            }
        combined[doc_id]["bm25_rank"] = i

    for i, result in enumerate(semantic_results, 1):
        doc_id = result["id"]
        if doc_id not in combined:
            combined[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": 0,
                "semantic_rank": 0,
            }
        combined[doc_id]["semantic_rank"] = i

    hybrid_results = []
    for doc_id, data in combined.items():
        total_rrf = 0
        if data["bm25_rank"] != 0:
            total_rrf += rrf_score(data["bm25_rank"], k)
        if data["semantic_rank"] != 0:
            total_rrf += rrf_score(data["semantic_rank"], k)

        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=total_rrf,
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)
