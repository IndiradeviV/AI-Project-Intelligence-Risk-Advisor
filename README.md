# AI Project Intelligence & Risk Advisor

## Project Overview

The **AI Project Intelligence & Risk Advisor** is an AI-powered project management application developed using Python and Streamlit. It helps software development teams and student project groups analyze project documents, identify potential risks, track project issues, and understand overall project health.

Project information is often scattered across proposals, meeting notes, sprint updates, task lists, and progress reports. Manually analyzing these documents can be time-consuming and may result in missed risks, unclear responsibilities, and delayed decisions.

This project uses **Retrieval-Augmented Generation (RAG), semantic search, and specialized AI agents** to extract meaningful information from project documents and provide useful project insights.

## Objectives

- Upload and process project documents in multiple formats.
- Extract and organize information from project documents.
- Generate text embeddings for semantic representation.
- Store document chunks and embeddings in a persistent vector database.
- Retrieve relevant information using semantic similarity search.
- Extract project scope, objectives, and deliverables.
- Identify potential project risks and blockers.
- Generate project-related documentation and insights.
- Assess project health using relevant project indicators.
- Provide an AI assistant for project-related queries.

## Key Features

### 1. Document Upload and Processing
- Supports PDF, DOCX, CSV, and TXT documents.
- Extracts readable text from uploaded files.
- Splits extracted text into smaller overlapping chunks.
- Prepares document content for embedding generation and retrieval.

### 2. Semantic Search and Retrieval
- Uses Sentence Transformers to generate text embeddings.
- Stores document chunks and embeddings in ChromaDB.
- Retrieves relevant passages based on user queries.
- Supports persistent vector storage.

### 3. Scope Extraction Agent
- Analyzes project documents to identify project scope.
- Extracts objectives, requirements, and deliverables.
- Helps understand the expected project outcomes.

### 4. Risk Detection Agent
- Analyzes project information to identify potential risks.
- Highlights issues that may affect project execution.
- Provides risk-related insights and recommendations.

### 5. Blocker and Action Item Agent
- Identifies project blockers and pending activities.
- Extracts action items from project information.
- Helps teams identify issues that require attention.

### 6. Documentation Agent
- Uses project information to support structured documentation.
- Helps organize relevant project details into useful outputs.

### 7. Project Intelligence Agent
- Supports intelligent analysis of project-related information.
- Uses project context to provide relevant insights.

### 8. Project Health Assessment
- Evaluates project health using scope, timeline, and blocker-related indicators.
- Calculates an overall project health score.
- Presents assessment results through the application.

### 9. AI Project Assistant
- Allows users to ask questions about project information.
- Uses relevant project context to support responses.
- Provides a conversational way to access project insights.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Pretrained embedding model |
| ChromaDB | Vector database and similarity search |
| Groq LLM | AI-powered analysis and response generation |
| LangChain Text Splitters | Text chunking |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| pandas | CSV processing |
| pytest | Automated testing |

## System Architecture

The application follows a document-processing and multi-agent RAG workflow.

1. **Document Upload:** Users upload project documents in supported formats.
2. **Text Extraction:** The document loader extracts readable text.
3. **Text Chunking:** Extracted text is divided into smaller overlapping chunks.
4. **Embedding Generation:** The embedding model converts text chunks into numerical vectors.
5. **Vector Storage:** ChromaDB stores embeddings, document chunks, and associated metadata.
6. **Query Processing:** User queries are processed to identify relevant project information.
7. **Semantic Retrieval:** ChromaDB retrieves relevant document chunks.
8. **AI Agent Analysis:** Specialized agents analyze project information for scope, risks, blockers, and documentation.
9. **Project Health Assessment:** Project indicators are used to calculate an overall project health score.
10. **Result Presentation:** The application displays project insights and AI-assisted responses through the Streamlit interface.

## Project Structure

```text
AI_Project_Risk_Advisor/
│
├── agents/
│   ├── __init__.py
│   ├── blocker_action_agent.py
│   ├── documentation_agent.py
│   ├── project_health_agent.py
│   ├── project_intelligence_agent.py
│   ├── risk_detection_agent.py
│   └── scope_extraction_agent.py
│
├── database/
│   └── chroma_store.py
│
├── docs/
│
├── embeddings/
│   └── embedding_model.py
│
├── ingestion/
│
├── models/
│
├── tests/
│   ├── test_assistant_accuracy.py
│   ├── test_blocker_agent.py
│   ├── test_chroma.py
│   ├── test_chunker.py
│   ├── test_documentation_agent.py
│   ├── test_embeddings.py
│   ├── test_loader.py
│   ├── test_project_health.py
│   ├── test_project_intelligence.py
│   ├── test_risk_agent.py
│   └── test_scope_agent.py
│
├── app.py
├── check_database.py
├── check_groq_key.py
├── check_project_ids.py
├── label_ai_risk_document.py
├── sample_ai_risk_advisor.txt
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

*Note: This structure reflects the files and folders visible in the project workspace. The contents of collapsed folders may include additional files.*

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- Git
- Visual Studio Code or another Python-compatible editor
- Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/IndiradeviV/AI-Project-Intelligence-Risk-Advisor.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-Project-Intelligence-Risk-Advisor
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory and configure your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

**Important:** Never upload your `.env` file or expose API keys and other credentials on GitHub.

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will start locally, and Streamlit will display a URL in the terminal. Open that URL in your browser.

## Running Tests

To execute the automated test cases, run:

```bash
python -m pytest tests/ -v
```

The test suite includes tests for:

- Document loading
- Text chunking
- Embedding generation
- ChromaDB operations
- Scope extraction
- Risk detection
- Blocker and action item identification
- Documentation generation
- Project health assessment
- Project intelligence and assistant accuracy

Review the test output to verify the current test results.

## Project Benefits

- Reduces manual effort in reviewing project documents.
- Improves visibility into project scope and deliverables.
- Helps identify potential risks and blockers.
- Supports project health monitoring.
- Organizes project information in a centralized application.
- Assists teams in making informed project decisions.

## Limitations

- Analysis quality depends on the quality and completeness of uploaded documents.
- AI-generated insights may require human verification.
- Project health assessments depend on the available project information and analysis results.
- The application is designed to support project management decisions, not replace human judgment.

## Future Enhancements

- Integration with external project management platforms.
- Advanced project schedule and dependency analysis.
- Historical project performance analysis.
- Improved risk prediction and prioritization.
- Automated progress tracking.
- Enhanced retrieval evaluation and response accuracy.

## Author

Indira Devi V
