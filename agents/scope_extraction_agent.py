import json
import os

from dotenv import load_dotenv
from groq import Groq

from database.chroma_store import ChromaStore
from embeddings.embedding_model import EmbeddingModel


class ScopeExtractionAgent:
    """Extract project scope and deliverables from stored documents."""

    def __init__(self):
        load_dotenv(override=True)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Check your local .env file."
            )

        self.client = Groq(api_key=api_key)
        self.embedding_model = EmbeddingModel()
        self.chroma_store = ChromaStore()

    def extract_scope(self, query: str, top_k: int = 8) -> dict:
        """Retrieve relevant project context and extract structured details."""

        query_embedding = self.embedding_model.embed_query(query)

        results = self.chroma_store.search(
    query_embedding=query_embedding,
    top_k=top_k,
    project_id="ai_project_intelligence_risk_advisor"
)

        documents = results.get("documents", [])
        chunks = documents[0] if documents else []

        if not chunks:
            return {
                "status": "no_context",
                "message": (
                    "No indexed project documents were found. "
                    "Upload and index project documents first."
                )
            }

        context = "\n\n".join(
            f"[Document chunk {index + 1}]\n{chunk}"
            for index, chunk in enumerate(chunks)
        )

        prompt = f"""
You are a Scope and Deliverable Extraction Agent
for a project intelligence system.

Extract information only from the supplied document context.
Do not invent missing dates, people, milestones, or deliverables.
If information is unavailable, write "Not specified".
Distinguish explicit facts from reasonable inferences.

Return valid JSON with these keys:
{{
  "project_goals": [],
  "deliverables": [],
  "milestones": [],
  "timelines": [],
  "responsibilities": [],
  "missing_information": []
}}

For each item, include a short description and supporting evidence
from the context. Do not use outside knowledge.

DOCUMENT CONTEXT:
{context}

USER REQUEST:
{query}
"""

        response = self.client.chat.completions.create(
           model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract project information accurately. "
                        "Return only valid JSON."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content

        try:
            extracted_data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": "invalid_response",
                "raw_response": content
            }

        return {
            "status": "success",
            "source_chunks": len(chunks),
            "extracted_data": extracted_data
        }
