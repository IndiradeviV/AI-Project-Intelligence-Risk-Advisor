
from embeddings.embedding_model import EmbeddingModel


def main():
    embedder = EmbeddingModel()

    texts = [
        "The project is progressing according to schedule.",
        "The project deadline may be delayed."
    ]

    vectors = embedder.embed_texts(texts)

    print("Number of vectors:", len(vectors))
    print("Vector dimensions:", len(vectors[0]))
    print("Embedding generated successfully!")


if __name__ == "__main__":
    main()
