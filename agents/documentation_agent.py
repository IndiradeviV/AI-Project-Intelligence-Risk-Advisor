
import json
import os

from dotenv import load_dotenv
from groq import Groq

from database.chroma_store import ChromaStore
from embeddings.embedding_model import EmbeddingModel


class DocumentationAgent:
    """Generate structured project documentation from indexed documents."""

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

    def generate_documentation(
        self,
        query: str,
        top_k: int = 8
    ) -> dict:
        """Generate user stories, risk register, and action items."""

        if not query or not query.strip():
            raise ValueError("Please provide a documentation request.")

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
Generate structured project documentation from the supplied evidence.

USER REQUEST:
{query}

PROJECT DOCUMENTS:
{context}

Return ONLY valid JSON using this structure:
{{
  "user_stories": [
    {{
      "title": "Short title",
      "story": "As a [user], I want [feature], so that [benefit].",
      "acceptance_criteria": ["Specific testable criterion"],
      "evidence": "Source evidence or Clearly labeled recommendation"
    }}
  ],
  "risk_register": [
    {{
      "risk": "Risk description",
      "likelihood": "High, Medium, Low, or Unknown",
      "impact": "High, Medium, Low, or Unknown",
      "mitigation": "Mitigation recommendation",
      "evidence": "Supporting source evidence"
    }}
  ],
  "action_items": [
    {{
      "action": "Action description",
      "owner": "Documented owner or Not specified",
      "deadline": "Documented deadline or Not specified",
      "priority": "High, Medium, Low, or Unknown",
      "evidence": "Supporting source evidence or Clearly labeled recommendation"
    }}
  ],
  "missing_information": [
    "Information needed to complete the documentation"
  ]
}}

Rules:
- Do not invent project facts, owners, deadlines, or status.
- Clearly distinguish source facts from suggested user stories and recommendations.
- Acceptance criteria may be proposed and should be labeled as recommendations
  when not explicitly documented.
- Use Unknown or Not specified when evidence is insufficient.
- Return empty lists if no supported items can be produced.
- Return valid JSON only, without Markdown fences.
"""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a project documentation specialist. "
                        "Use supplied evidence carefully and return valid JSON only."
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
            documentation = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": "invalid_response",
                "raw_response": content
            }

        return {
            "status": "success",
            "source_chunks": len(chunks),
            "documentation": documentation
        }
