
from database.chroma_store import ChromaStore

store = ChromaStore()

records = store.collection.get(
    limit=100,
    include=["metadatas"]
)

print("Total indexed chunks:", store.count_chunks())
print("\nProject ID counts:")

counts = {}

for metadata in records.get("metadatas", []):
    metadata = metadata or {}
    project_id = metadata.get("project_id", "UNASSIGNED")
    counts[project_id] = counts.get(project_id, 0) + 1

for project_id, count in counts.items():
    print(f"{project_id}: {count} chunks")
