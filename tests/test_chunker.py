
from ingestion.chunker import split_text


def test_chunking():
    text = (
        "WeatherGPT is a weather analysis project. " * 100
    )

    chunks = split_text(
        text,
        chunk_size=200,
        chunk_overlap=30
    )

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)

    print("Chunking successful!")
    print("Total chunks:", len(chunks))
    print("\nFirst chunk preview:")
    print(chunks[0])


if __name__ == "__main__":
    test_chunking()
