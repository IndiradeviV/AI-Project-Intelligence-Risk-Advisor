
import json
import os

from dotenv import load_dotenv
from groq import Groq

from database.chroma_store import ChromaStore
from embeddings.embedding_model import EmbeddingModel


class BlockerActionAgent:
    """Identify project blockers and actionable next steps."""

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

    def identify_blockers(
        self,
        query: str,
        top_k: int = 8
    ) -> dict:
        """Retrieve project-specific context and identify blockers."""

        if not query or not query.strip():
            raise ValueError("Please provide a blocker analysis query.")

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
                    "No indexed documents were found for the "
                    "AI Project Intelligence & Risk Advisor."
                )
            }

        context = "\n\n".join(
            f"[Document chunk {index + 1}]\n{chunk}"
            for index, chunk in enumerate(chunks)
        )

        prompt = f"""
Identify project blockers and action items using only the
evidence in the supplied project documents.

USER REQUEST:
{query}

PROJECT DOCUMENTS:
{context}

Return ONLY valid JSON in this structure:
{{
  "blockers": [
    {{
      "blocker": "Description of the blocker",
      "evidence": "Supporting text from the documents",
      "impact": "Known impact or Unknown",
      "affected_deliverable": "Deliverable or Unknown",
      "suggested_resolution": "Recommended way to address it"
    }}
  ],
  "action_items": [
    {{
      "action": "Specific next action",
      "owner": "Documented owner or Not specified",
      "deadline": "Documented deadline or Not specified",
      "priority": "High, Medium, Low, or Unknown",
      "evidence": "Supporting text, or label this as a recommendation"
    }}
  ],
  "unresolved_dependencies": [
    {{
      "dependency": "Dependency description",
      "evidence": "Supporting document evidence",
      "next_step": "Suggested next step"
    }}
  ],
  "missing_information": [
    "Information needed to improve blocker or action-item tracking"
  ]
}}

Rules:
- Do not invent owners, deadlines, task statuses, or facts.
- Distinguish documented action items from your recommendations.
- If an owner or deadline is absent, use Not specified.
- If evidence is insufficient, say so.
- Return empty lists when no supported items are found.
- Return valid JSON only, without Markdown fences.
"""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful project coordinator. "
                        "Ground blocker and action-item analysis in "
                        "the supplied documents. Return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
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
            "blocker_analysis": extracted_data
        }
