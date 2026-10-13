
import hashlib
from pathlib import Path

import chromadb


class ChromaStore:
    """Manage persistent storage of document chunks and embeddings."""

    def __init__(self, persist_directory: str = "chroma_db"):
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="project_documents",
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def create_document_id(
        filename: str,
        file_bytes: bytes
    ) -> str:
        """Create a stable ID for a particular file version."""
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        return f"{filename}_{content_hash[:16]}"

    def add_chunks(
        self,
        filename: str,
        file_bytes: bytes,
        chunks: list[str],
        embeddings: list[list[float]]
    ) -> int:
        """Store chunks, vectors, and metadata in ChromaDB."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        document_id = self.create_document_id(
            filename,
            file_bytes
        )

        ids = [
            f"{document_id}_chunk_{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "file_type": Path(filename).suffix.lower(),
                "chunk_index": index,
                "character_count": len(chunk)
            }
            for index, chunk in enumerate(chunks)
        ]

        # Upsert prevents duplicate records when the same file is processed again.
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    def count_chunks(self) -> int:
        """Return the total number of stored chunks."""
        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ) -> dict:
        """Retrieve chunks similar to a query embedding."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.collection.count() == 0:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
