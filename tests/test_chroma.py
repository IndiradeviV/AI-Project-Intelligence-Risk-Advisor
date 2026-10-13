
from database.chroma_store import ChromaStore


def main():
    store = ChromaStore(
        persist_directory="test_chroma_db"
    )

    filename = "sample.txt"
    file_bytes = b"Sample project document"

    chunks = [
        "The project is progressing according to schedule.",
        "The team completed the initial design."
    ]

    # Two sample vectors with the same dimensions.
    # These are test vectors, not model-generated embeddings.
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    added = store.add_chunks(
        filename,
        file_bytes,
        chunks,
        embeddings
    )

    print("Chunks submitted:", added)
    print("Total chunks in database:", store.count_chunks())

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2
    )

    print("Retrieved documents:", results["documents"])


if __name__ == "__main__":
    main()
