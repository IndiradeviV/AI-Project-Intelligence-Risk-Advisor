
# AI Project Intelligence & Risk Advisor
## Study Notes — Milestone 1

# 1. Retrieval-Augmented Generation (RAG)

## 1.1 Definition

Retrieval-Augmented Generation (RAG) is an approach that combines information retrieval with a language model. Relevant information is retrieved from an external knowledge source and provided to the language model to help it generate a grounded response.

## 1.2 Why RAG Is Useful

- It allows a system to retrieve information from project-specific documents.
- It can provide answers grounded in supplied documents.
- Documents can be indexed without retraining a language model.
- Retrieved passages can help users verify where information came from.

## 1.3 Main Components

### Document Ingestion

Accepts source documents and extracts their text.

In this project, the supported formats are PDF, DOCX, CSV, and TXT.

### Text Chunking

Splits extracted text into smaller sections. Chunking helps process long documents and retrieve focused passages.

The current implementation uses overlapping character-based chunks.

### Embedding Generation

Converts text into numerical vectors that represent semantic information.

This project uses the Sentence Transformers model `all-MiniLM-L6-v2`.

### Vector Database

Stores embeddings, text chunks, and metadata for retrieval.

This project uses ChromaDB with persistent local storage.

### Retrieval

Converts a query into an embedding and searches for similar stored vectors.

The current application displays retrieved text chunks. It does not yet generate answers using a language model.

## 1.4 RAG Workflow

1. Upload a document.
2. Extract its text.
3. Split the text into chunks.
4. Generate an embedding for each chunk.
5. Store the chunks and embeddings in a vector database.
6. Convert a user query into an embedding.
7. Retrieve relevant chunks using similarity search.
8. In a complete RAG system, pass retrieved context to a language model to generate a response.

## 1.5 RAG in This Project

The current implementation covers document ingestion, chunking, embeddings, persistent vector indexing, and basic retrieval.

Language-model-based answer generation is a future extension.

# 2. Multi-Agent Design Patterns

## 2.1 Definition

A multi-agent system contains multiple agents that perform different tasks and may coordinate to achieve a larger objective.

An agent may use instructions, tools, context, and a language model to perform its assigned responsibility.

## 2.2 Why Use Multiple Agents?

- Separate complex work into smaller responsibilities.
- Allow specialized processing for different tasks.
- Make components easier to test and maintain.
- Support coordination between analysis and decision-support tasks.

## 2.3 Common Design Patterns

### Sequential Pattern

Agents or components execute in a defined order. The output of one step becomes the input to the next step.

Example: document extraction → chunking → embedding generation → indexing.

The current document pipeline follows a sequential processing flow, but these stages are ordinary software components, not autonomous AI agents.

### Coordinator Pattern

A central coordinator assigns tasks to specialized agents and combines their outputs.

Possible future use: a project-analysis coordinator could request information from document analysis and schedule analysis components.

### Parallel Pattern

Independent tasks run separately and their results are combined.

Possible future use: different analysis agents could independently examine project scope, schedule, and resource information.

### Specialist Pattern

Each agent focuses on a specific domain or task.

Possible future use:
- Document Analysis Agent
- Schedule Analysis Agent
- Risk Assessment Agent
- Recommendation Agent

These are proposed future roles, not implemented agents in the current application.

## 2.4 Multi-Agent Design Considerations

- Define clear responsibilities for each agent.
- Specify the inputs and outputs of each task.
- Avoid unnecessary duplication between agents.
- Handle failures and incomplete information.
- Validate results before combining them.
- Consider execution cost, latency, and maintainability.

## 2.5 Application to This Project

The current milestone establishes the document-processing foundation. Multi-agent coordination and automated risk analysis can be introduced in a later phase if required by the project plan.

# 3. Project Management Fundamentals

## 3.1 Project Scope

Project scope defines the work and deliverables included in a project.

For the current milestone, the scope includes:
- Studying the required concepts.
- Designing system architecture and data models.
- Supporting multiple document formats.
- Implementing chunking and embedding generation.
- Indexing document vectors in ChromaDB.

## 3.2 Work Breakdown Structure

A Work Breakdown Structure (WBS) divides a project into smaller, manageable tasks.

Example:

1. Requirements and technology study.
2. Architecture and data-model design.
3. Document ingestion implementation.
4. Chunking implementation.
5. Embedding generation.
6. ChromaDB indexing.
7. Retrieval testing.
8. Documentation and validation.

## 3.3 Milestones

A milestone is a checkpoint used to track progress or confirm completion of a major stage.

Milestone 1 establishes the design and document retrieval foundation for later project-intelligence features.

## 3.4 Dependencies

Some tasks depend on earlier tasks.

For example:
- Chunking depends on extracted text.
- Embedding generation depends on text chunks.
- Vector indexing depends on chunks and embeddings.
- Similarity retrieval depends on an indexed collection and query embeddings.

## 3.5 Risk Management

Project risk management involves identifying, assessing, responding to, and monitoring uncertainties that may affect project objectives.

Common categories include:
- Schedule risk
- Resource risk
- Technical risk
- Scope risk
- Data-quality risk

The current application does not automatically predict these risks. Automated risk analysis is a potential future feature.

## 3.6 Progress Monitoring

Progress can be monitored by comparing completed tasks against planned deliverables.

For this milestone, useful checks include:
- Whether all required file formats can be processed.
- Whether chunks are generated correctly.
- Whether embeddings have the expected dimensions.
- Whether ChromaDB stores and retrieves indexed chunks.
- Whether the documentation matches the implementation.

# 4. Summary

This milestone combines three areas of knowledge:

- RAG provides the overall retrieval-based AI architecture.
- Multi-agent design patterns explain how future specialized AI tasks may be coordinated.
- Project management fundamentals support planning, task decomposition, dependency management, and progress tracking.

The current implementation focuses on document ingestion and retrieval. More advanced AI analysis and multi-agent functionality remain future extensions.
