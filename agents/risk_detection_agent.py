import json
import os

from dotenv import load_dotenv
from groq import Groq

from database.chroma_store import ChromaStore
from embeddings.embedding_model import EmbeddingModel


class RiskDetectionAgent:
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

    def detect_risks(self, query: str, top_k: int = 8) -> dict:
        if not query or not query.strip():
            raise ValueError("Please provide a project risk query.")

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
            f"[Document chunk {i + 1}]\n{chunk}"
            for i, chunk in enumerate(chunks)
        )

        prompt = f"""
Analyze the project documents below to identify evidence-based
project risks and delivery challenges.

USER REQUEST:
{query}

PROJECT DOCUMENTS:
{context}

Return ONLY valid JSON with this structure:
{{
  "risks": [
    {{
      "risk": "Short risk description",
      "category": "Schedule, Dependency, Resource, Scope, or Other",
      "likelihood": "High, Medium, Low, or Unknown",
      "impact": "High, Medium, Low, or Unknown",
      "severity": "High, Medium, Low, or Unknown",
      "evidence": "Specific supporting text from the documents",
      "mitigation": "Practical suggested action"
    }}
  ],
  "delivery_forecast": {{
    "status": "At Risk, On Track, Delayed, or Insufficient Information",
    "reason": "Explain the assessment using document evidence"
  }},
  "dependency_gaps": [
    {{
      "dependency": "Dependency or missing prerequisite",
      "evidence": "Supporting document evidence",
      "recommended_action": "Suggested next action"
    }}
  ],
  "missing_information": [
    "Information needed for a more reliable assessment"
  ]
}}

Rules:
- Never invent dates, owners, project progress, or facts.
- Distinguish documented facts from potential risks.
- Do not claim a deadline will be missed unless the evidence supports it.
- If likelihood or impact cannot be assessed, use Unknown.
- If there is insufficient evidence for a delivery forecast,
  use Insufficient Information.
- Suggested mitigations are recommendations, not documented facts.
- If no risks are supported, return an empty risks list.
- Return valid JSON only, without Markdown fences.
"""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful project risk analyst. "
                        "Ground assessments in supplied evidence and "
                        "return only valid JSON."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content

        try:
            risk_data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": "invalid_response",
                "raw_response": content
            }

        return {
            "status": "success",
            "source_chunks": len(chunks),
            "risk_analysis": risk_data
        }
