from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", docs=[]):
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.texts = []

        for doc in self.docs:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)

        results = []
        for i, txt in enumerate(self.text_embeddings):
            doc = self.docs[i]
            similarity = cosine_similarity(image_embedding, txt)
            results.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity_score": similarity,
                }
            )

        sorted_results = sorted(
            results, key=lambda x: x["similarity_score"], reverse=True
        )
        return sorted_results[:5]

    def embed_image(self, image_path: str):
        img = Image.open(image_path)
        embedding = self.model.encode([img], show_progress_bar=True)  # type: ignore[arg-type]
        return embedding[0]


def verify_image_embedding(image_path: str):
    multimodal_search = MultimodalSearch()
    image_embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    movies = load_movies()
    multimodal_search = MultimodalSearch(docs=movies)
    return multimodal_search.search_with_image(image_path)
