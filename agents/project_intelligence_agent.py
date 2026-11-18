
import os

from dotenv import load_dotenv
from groq import Groq

from database.chroma_store import ChromaStore
from embeddings.embedding_model import EmbeddingModel


class ProjectIntelligenceAgent:
    """Answer project-related questions using RAG."""

    def __init__(self):

        load_dotenv(override=True)

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Check your .env file."
            )

        self.client = Groq(api_key=api_key)

        self.embedding_model = EmbeddingModel()

        self.chroma_store = ChromaStore(
            persist_directory="chroma_db"
        )

    def ask_question(self, question: str, top_k: int = 5) -> dict:
        """Retrieve project context and generate an answer."""

        if not question or not question.strip():
            raise ValueError("Please enter a project-related question.")

        # 1. Convert question into embedding
        query_embedding = self.embedding_model.embed_query(question)

        # 2. Retrieve relevant project document chunks
        results = self.chroma_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            project_id="ai_project_intelligence_risk_advisor"
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return {
                "status": "no_context",
                "message": (
                    "No relevant project information was found. "
                    "Please upload and index project documents."
                )
            }

        # 3. Prepare context and source information
        context_parts = []
        sources = []

        for index, document in enumerate(documents):

            metadata = metadatas[index] or {}

            filename = metadata.get(
                "filename",
                "Unknown document"
            )

            chunk_index = metadata.get(
                "chunk_index",
                index
            )

            context_parts.append(
                f"[Source: {filename}, Chunk: {chunk_index + 1}]\n"
                f"{document}"
            )

            source = {
                "filename": filename,
                "chunk": chunk_index + 1
            }

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        # 4. Generate grounded answer using Groq
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent project management "
                        "assistant. Answer questions using only the "
                        "provided project document context. "
                        "Do not invent project facts, task statuses, "
                        "owners, deadlines, or dates. "
                        "If information is missing, clearly state "
                        "that it is not specified in the documents. "
                        "Distinguish documented facts from recommendations. "
                        "Mention relevant source filenames when possible."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
PROJECT DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Instructions:
1. Answer clearly and directly.
2. Use only the supplied project context.
3. Do not assume missing information.
4. Mention relevant document names.
5. If the documents do not contain the answer, say so.
"""
                }
            ],
            temperature=0.1
        )

        answer = response.choices[0].message.content

        if not answer:
            return {
                "status": "invalid_response",
                "message": "The AI assistant returned an empty response."
            }

        # 5. Return answer and source references
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "source_chunks": len(documents),
            "sources": sources
        }
