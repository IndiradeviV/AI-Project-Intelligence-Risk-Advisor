
import chromadb

# Connect to the existing persistent database
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection_name = "project_documents"

# Check whether the collection exists
existing_collections = [
    collection.name for collection in client.list_collections()
]

if collection_name in existing_collections:
    # Delete the existing collection and all its records
    client.delete_collection(name=collection_name)
    print("Old project documents cleared successfully.")
else:
    print("No existing collection found.")

# Create a fresh empty collection
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

print("Fresh ChromaDB collection created.")
print("Current document chunk count:", collection.count())
