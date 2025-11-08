import os

from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import HybridSearch
from lib.search_utils import (DEFAULT_K_VALUE, DEFAULT_SEARCH_LIMIT,
                              SEARCH_MULTIPLIER, load_movies)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def question_command(query: str) -> dict:
    return question(query)


def question(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, DEFAULT_K_VALUE, limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    question_answer = generate_question_answer(search_results, query, limit)
    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": question_answer,
    }


def generate_question_answer(
    search_results: list[dict], query: str, limit: int = 5
) -> str:
    docs = ""

    for result in search_results[:limit]:
        docs += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def citations_command(query: str) -> dict:
    return citations(query)


def citations(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, DEFAULT_K_VALUE, limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    citations = generate_citations(search_results, query, limit)
    return {
        "query": query,
        "search_results": search_results[:limit],
        "citations": citations,
    }


def generate_citations(search_results: list[dict], query: str, limit: int = 5) -> str:
    docs = ""

    for result in search_results[:limit]:
        docs += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def summarize_command(query: str):
    return summarize(query)


def summarize(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, DEFAULT_K_VALUE, limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    summary = generate_summary(search_results, query, limit)
    return {
        "query": query,
        "search_results": search_results[:limit],
        "summary": summary,
    }


def generate_summary(search_results: list[dict], query: str, limit: int = 5) -> str:
    docs = ""

    for result in search_results[:limit]:
        docs += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def rag_command(query: str):
    return rag(query)


def rag(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, DEFAULT_K_VALUE, limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {"query": query, "search_results": [], "error": "No results found"}

    answer = generate_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer,
    }


def generate_answer(search_results: list[dict], query: str, limit: int = 5) -> str:
    docs = ""

    for result in search_results[:limit]:
        docs += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()
