
from database.chroma_store import ChromaStore

store = ChromaStore()

results = store.search(
    query_embedding=[0.0] * 384,
    top_k=3
)

documents = results.get("documents", [])

print("Total indexed chunks:", store.count_chunks())
print("\nSample stored chunks:\n")

if documents and documents[0]:
    for index, chunk in enumerate(documents[0], start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk[:700])
else:
    print("No documents returned by the search.")
