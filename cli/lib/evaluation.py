import json
import os

from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import HybridSearch
from lib.search_utils import DEFAULT_K_VALUE, load_golden_dataset, load_movies
from lib.semantic_search import SemanticSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def llm_evaluation(query: str, rrf_results: list[dict]) -> list[int]:
    formatted_results = []
    for i, result in enumerate(rrf_results, 1):
        formatted_results.append(f"{i}. {result['title']}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(model=model, contents=prompt)
    cleaned_response = (response.text or "").strip()
    json_response = json.loads(cleaned_response)

    if len(json_response) == len(rrf_results):
        return list(map(int, json_response))

    raise ValueError(
        f"LLM response parsing error: Expected {len(rrf_results)} scores, got {len(json_response)}. Response: {json_response}"
    )


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    retrieved_relevant = 0
    for doc in retrieved_docs[:k]:
        if doc in relevant_docs:
            retrieved_relevant += 1
    return retrieved_relevant / len(relevant_docs)


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k


def evaluate_result(limit: int = 5) -> dict:
    movies = load_movies()
    golden_data = load_golden_dataset()
    test_cases = golden_data["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0
    total_recall = 0
    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=DEFAULT_K_VALUE, limit=limit)
        retrieved_docs = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        f1 = 2 * (precision * recall) / (precision + recall)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }

        total_precision += precision
        total_recall += recall

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }
