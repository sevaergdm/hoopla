from cli.lib.inverted_index import InvertedIndex
from .search_utils import (DEFAULT_SEARCH_LIMIT, load_movies,
                           tokenize)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list:
    movies = load_movies()
    results = []

    for movie in movies:
        movie_tokens = tokenize(movie.get("title", ""))
        query_tokens = tokenize(query)

        if has_matching_token(query_tokens, movie_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for title_token in title_tokens:
        for query_token in query_tokens:
            if query_token in title_token:
                return True
    return False
