# AI Project Intelligence & Risk Advisor

## Project Overview

The **AI Project Intelligence & Risk Advisor** is a document-processing and semantic-search application built with Python and Streamlit. It allows users to upload project documents and retrieve relevant information using text embeddings and vector similarity search.

This project implements the foundational document ingestion and retrieval components of a Retrieval-Augmented Generation (RAG) pipeline.

## Objectives

- Upload and process documents in multiple file formats.
- Extract text and divide it into smaller, manageable chunks.
- Generate text embeddings for semantic representation.
- Store document chunks and embeddings in a persistent vector database.
- Retrieve relevant passages based on user queries.

## Features

- **Multi-format document upload:** PDF, DOCX, CSV, and TXT.
- **Text extraction:** Extracts readable text from supported documents.
- **Text chunking:** Splits extracted text into overlapping chunks.
- **Embedding generation:** Uses Sentence Transformers to convert text into numerical vectors.
- **Vector storage:** Stores embeddings and metadata using ChromaDB.
- **Semantic search:** Retrieves relevant document passages based on a search query.
- **Persistent storage:** Retains the local ChromaDB collection between application sessions.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Pretrained embedding model |
| ChromaDB | Vector database and similarity search |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| pandas | CSV processing |
| LangChain Text Splitters | Text chunking |
| pytest | Automated testing |

## System Architecture

The application follows this workflow:

1. **Document Upload:** The user uploads one or more supported files.
2. **Text Extraction:** The document loader extracts the text.
3. **Text Chunking:** The extracted text is split into smaller overlapping chunks.
4. **Embedding Generation:** The embedding model converts each chunk into a vector.
5. **Vector Storage:** ChromaDB stores the vectors, text chunks, and associated metadata.
6. **Query Processing:** The user's search query is converted into an embedding.
7. **Similarity Search:** ChromaDB retrieves the closest matching document chunks.
8. **Result Display:** The application displays the retrieved passages and their metadata.

## Project Structure

```text
AI_Project_Risk_Advisor/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── database/
│   └── chroma_store.py
├── docs/
│   ├── architecture.md
│   └── data_models.md
├── embeddings/
│   └── embedding_model.py
├── ingestion/
│   ├── document_loader.py
│   └── chunker.py
└── tests/
    ├── test_chroma.py
    ├── test_chunker.py
    ├── test_embeddings.py
    └── test_loader.py
```

The `chroma_db/` directory is generated locally when the application stores data. It is excluded from GitHub using `.gitignore`.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- Git
- Visual Studio Code or another Python-compatible editor

### 1. Clone the repository

```bash
git clone https://github.com/IndiradeviV/AI-Project-Intelligence-Risk-Advisor.git
```

### 2. Open the project directory

```bash
cd AI-Project-Intelligence-Risk-Advisor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

## Running Tests

To execute the automated tests, run:

```bash
python -m pytest tests/ -v
```

The tests cover the document-loading, chunking, embedding, and ChromaDB components. Review the test output to confirm the current results.

## Current Scope

This milestone focuses on document ingestion, embedding generation, persistent vector storage, and semantic retrieval.

The application retrieves relevant passages from uploaded documents. It does not currently generate answers using a large language model or perform fully automated project-risk prediction.

## Limitations and Future Enhancements

Potential future enhancements include:

- LLM-based answer generation using retrieved document context.
- Project milestone and dependency analysis.
- Identification of project risks and potential mitigation strategies.
- Risk scoring and prioritization.
- Project progress summaries and actionable recommendations.
- Improved retrieval evaluation and relevance tuning.

These are proposed enhancements and are not claimed as implemented features.

## Documentation

Additional technical documentation is available in the `docs/` directory:

- `architecture.md` — system architecture and workflow.
- `data_models.md` — document, chunk, embedding, and metadata structures.

## Author

Developed as an academic project for Milestone 1.

