
from database.chroma_store import ChromaStore

store = ChromaStore()

target_filename = "sample_ai_risk_advisor.txt"
project_id = "ai_project_intelligence_risk_advisor"

# Find only chunks belonging to the sample document.
records = store.collection.get(
    where={"filename": target_filename},
    include=["metadatas"]
)

ids = records.get("ids", [])
metadatas = records.get("metadatas", [])

if not ids:
    print(f"No indexed chunks found for: {target_filename}")
    print("Check the uploaded filename in your Streamlit app.")
else:
    updated_metadatas = []

    for metadata in metadatas:
        updated = dict(metadata or {})
        updated["project_id"] = project_id
        updated_metadatas.append(updated)

    # Update metadata only. Document text and embeddings are preserved.
    store.collection.update(
        ids=ids,
        metadatas=updated_metadatas
    )

    print(f"Updated chunks: {len(ids)}")
    print(f"Project ID: {project_id}")
    print(f"Total database chunks: {store.count_chunks()}")
