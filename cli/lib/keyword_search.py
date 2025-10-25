from lib.inverted_index import InvertedIndex
from lib.search_utils import DEFAULT_SEARCH_LIMIT, tokenize


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
