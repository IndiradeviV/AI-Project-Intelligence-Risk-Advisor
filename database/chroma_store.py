import hashlib
from pathlib import Path

import chromadb


class ChromaStore:
    """Manage persistent storage of document chunks and embeddings."""

    def __init__(self, persist_directory: str = "chroma_db"):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="project_documents",
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def create_document_id(filename: str, file_bytes: bytes) -> str:
        """Create a stable ID for one exact file version."""
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        safe_filename = Path(filename).name
        return f"{safe_filename}_{content_hash[:16]}"

    def get_document_chunk_count(self, document_id: str) -> int:
        """Return the number of chunks stored for an exact document version."""
        result = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
        return len(result.get("ids", []))

    def add_chunks(
        self,
        filename: str,
        file_bytes: bytes,
        chunks: list[str],
        embeddings: list[list[float]],
        project_id: str = "default"
    ) -> int:
        """Add or refresh chunks without deleting other document versions."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )
        if not chunks:
            return 0

        document_id = self.create_document_id(filename, file_bytes)
        ids = [f"{document_id}_chunk_{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "filename": Path(filename).name,
                "file_type": Path(filename).suffix.lower(),
                "chunk_index": index,
                "character_count": len(chunk),
                "project_id": project_id
            }
            for index, chunk in enumerate(chunks)
        ]

        # Upsert uses stable IDs: re-uploading the same file version does not duplicate chunks.
        # A changed file has a different content hash and is stored as a new version.
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
        top_k: int = 3,
        project_id: str | None = None
    ) -> dict:
        """Retrieve similar chunks, optionally filtered by project."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        total_count = self.collection.count()
        if total_count == 0:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total_count),
            "include": ["documents", "metadatas", "distances"]
        }
        if project_id is not None:
            query_args["where"] = {"project_id": project_id}
        return self.collection.query(**query_args)
