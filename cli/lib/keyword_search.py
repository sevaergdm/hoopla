from lib.inverted_index import InvertedIndex
from lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT, tokenize


def bm25_search_command(query: str, limit: int = 5):
    idx = InvertedIndex()
    idx.load()
    search_results = idx.bm25_search(query, limit)
    output = {}
    for k, v in search_results.items():
        doc_id = k
        score = v
        movie_title = idx.docmap[doc_id]["title"]
        output[doc_id] = (movie_title, score)

    return output


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f"Unable to load index: {e}")
        return 0

    return idx.get_tf(doc_id, term)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f"Unable to load index: {e}")
        return []

    query_tokens = tokenize(query)
    seen = set()
    results = []

    for token in query_tokens:
        matches = idx.get_documents(token)
        for match in matches:
            if match in seen:
                continue
            seen.add(match)
            doc = idx.docmap.get(match, {})
            if not doc:
                continue
            results.append(doc)
            if len(results) >= limit:
                return results
    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for title_token in title_tokens:
        for query_token in query_tokens:
            if query_token in title_token:
                return True
    return False
