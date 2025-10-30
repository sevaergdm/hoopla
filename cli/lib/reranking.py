import os
from time import sleep

from dotenv import load_dotenv
from google import genai
from google.genai.files import json
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def rerank_cross_encoder(query: str, docs: list[dict], limit: int = 5) -> list[dict]:
    if not docs:
        return []

    pairs = []
    for doc in docs:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for i, score in enumerate(scores):
        docs[i]["cross_encoder_score"] = score

    sorted_docs = sorted(docs, key=lambda x: x["cross_encoder_score"], reverse=True) 

    return sorted_docs[:limit]


def rerank_batch(query: str, docs: list[dict], limit: int = 5) -> list[dict]:
    if not docs:
        return []

    doc_map = {}
    doc_list = []
    for doc in docs:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    response = client.models.generate_content(model=model, contents=prompt)
    cleaned_response = (response.text or "").strip()
    json_response = json.loads(cleaned_response)

    reranked = []
    for i, doc_id in enumerate(json_response):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    return reranked[:limit]


def rerank_individual(query: str, docs: list[dict], limit: int = 5) -> list[dict]:
    scored_docs = []

    for doc in docs:
        prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def rerank_result(
    query: str, docs: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    match method:
        case "cross_encoder":
            return rerank_cross_encoder(query, docs, limit)
        case "batch":
            return rerank_batch(query, docs, limit)
        case "individual":
            return rerank_individual(query, docs, limit)
        case _:
            return docs[:limit]
