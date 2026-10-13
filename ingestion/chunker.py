
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> list[str]:
    """
    Split extracted document text into smaller overlapping chunks.

    Args:
        text: Extracted document text.
        chunk_size: Maximum target size of each chunk in characters.
        chunk_overlap: Characters shared between adjacent chunks.

    Returns:
        A list of non-empty text chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_text(text)

    return [chunk.strip() for chunk in chunks if chunk.strip()]
