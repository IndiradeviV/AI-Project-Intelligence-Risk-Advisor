
# AI Project Intelligence & Risk Advisor
## System Architecture — Milestone 1

### 1. Overview

The AI Project Intelligence & Risk Advisor is designed to process project-related documents and prepare them for retrieval using a Retrieval-Augmented Generation (RAG) pipeline.

The current implementation focuses on document ingestion, text extraction, chunking, embedding generation, persistent vector storage, and similarity-based retrieval.

### 2. Technologies Used

- Python — core application logic
- Streamlit — user interface
- PyMuPDF — PDF text extraction
- python-docx — DOCX text extraction
- Pandas — CSV processing
- LangChain Text Splitters — text chunking
- Sentence Transformers — embedding generation
- ChromaDB — persistent vector database

### 3. System Architecture

```text
                 USER
                   |
                   v
          +------------------+
          |  Streamlit UI    |
          | Document Upload  |
          +------------------+
                   |
                   v
          +------------------+
          | Document Loader  |
          | PDF / DOCX / CSV |
          | TXT Extraction   |
          +------------------+
                   |
                   v
          +------------------+
          |   Text Chunker   |
          | Size: 350 chars  |
          | Overlap: 50 chars|
          +------------------+
                   |
                   v
          +------------------+
          | Embedding Model  |
          | all-MiniLM-L6-v2 |
          +------------------+
                   |
                   v
          +------------------+
          |    ChromaDB      |
          | Persistent Index |
          +------------------+
                   ^
                   |
          +------------------+
          | Query Embedding  |
          | Similarity Search|
          +------------------+
                   ^
                   |
          +------------------+
          | Search Interface |
          +------------------+
```

### 4. Component Responsibilities

#### 4.1 User Interface

File: `app.py`

Responsibilities:
- Accept multiple document uploads.
- Display document processing status.
- Show extracted text and chunk previews.
- Display the number of indexed chunks.
- Accept search queries and show retrieved results.

#### 4.2 Document Loader

File: `ingestion/document_loader.py`

Responsibilities:
- Extract text from PDF documents.
- Read paragraphs and tables from DOCX files.
- Convert CSV data into readable text.
- Read plain-text TXT files.
- Report empty or unsupported documents.

#### 4.3 Text Chunker

File: `ingestion/chunker.py`

Responsibilities:
- Split extracted text into smaller segments.
- Use configurable chunk size and overlap.
- Preserve some context between adjacent chunks.

Current search indexing configuration:
- Chunk size: 350 characters.
- Chunk overlap: 50 characters.

#### 4.4 Embedding Model

File: `embeddings/embedding_model.py`

Responsibilities:
- Convert document chunks into numerical vectors.
- Convert search queries into vectors using the same model.
- Produce 384-dimensional embeddings using `all-MiniLM-L6-v2`.

#### 4.5 Vector Database

File: `database/chroma_store.py`

Responsibilities:
- Store text chunks and their embeddings.
- Store document metadata such as filename and chunk index.
- Persist indexed data locally.
- Retrieve similar chunks for a query.

### 5. Document Processing Workflow

1. The user uploads one or more supported documents.
2. The loader extracts readable text.
3. The chunker splits the text into overlapping chunks.
4. The embedding model converts each chunk into a vector.
5. ChromaDB stores the chunks, vectors, and metadata.
6. The application displays processing statistics.

### 6. Query Retrieval Workflow

1. The user enters a search question.
2. The embedding model converts the question into a vector.
3. ChromaDB searches the stored vectors using cosine distance.
4. The application displays the retrieved text chunks and source metadata.
5. A relevance threshold can filter out less similar results.

### 7. Current Scope and Limitations

Implemented:
- Multi-format document ingestion.
- Text chunking.
- Embedding generation.
- Persistent vector indexing.
- Basic similarity-based retrieval.

Not yet implemented:
- LLM-generated answers.
- Multi-agent execution.
- Automated project-risk prediction.
- Proactive risk alerts.

These features are outside the current ingestion and retrieval implementation.

### 8. Conclusion

The current architecture establishes the document-processing and retrieval foundation for the AI Project Intelligence & Risk Advisor. It prepares project information for future AI-assisted analysis and decision-support capabilities.
