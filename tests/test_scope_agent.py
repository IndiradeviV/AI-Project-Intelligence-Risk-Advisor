from agents.scope_extraction_agent import ScopeExtractionAgent


def main():
    print("Initializing Scope Extraction Agent...")

    agent = ScopeExtractionAgent()

    print("Checking indexed document chunks...")

    count = agent.chroma_store.count_chunks()
    print(f"Total indexed chunks: {count}")

    if count == 0:
        print(
            "No indexed documents found. Start the app and upload "
            "a project document before running this test."
        )
        return

    query = (
        "Extract the project goals, deliverables, milestones, "
        "timelines, and responsibilities from the uploaded documents."
    )

    print("\nRunning extraction...")
    result = agent.extract_scope(query)

    print("\n--- Agent Result ---")
    print(result)


if __name__ == "__main__":
    main()
