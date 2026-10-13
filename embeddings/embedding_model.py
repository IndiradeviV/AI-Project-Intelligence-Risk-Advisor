
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Generate vector embeddings for document chunks and queries."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        # Load the model once when this class is initialized.
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of text chunks into embedding vectors."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert one search query into an embedding vector."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.embed_texts([query])[0]
