
import json

from agents.documentation_agent import DocumentationAgent


def main():
    print("=" * 60)
    print("DOCUMENTATION GENERATION AGENT - TEST")
    print("=" * 60)

    try:
        agent = DocumentationAgent()

        query = (
            "Generate project user stories with acceptance criteria, "
            "a risk register with mitigation recommendations, and "
            "an action-item summary with owners and deadlines where documented. "
            "List missing information."
        )

        print("\nGenerating project documentation...\n")

        result = agent.generate_documentation(query)

        print("TEST RESULT:")
        print(json.dumps(result, indent=4, ensure_ascii=False))

        if result.get("status") == "success":
            print("\nSUCCESS: Documentation generation completed.")
        elif result.get("status") == "no_context":
            print("\nNO CONTEXT: No project documents were retrieved.")
        else:
            print("\nCHECK REQUIRED: The agent returned invalid JSON.")

    except Exception as error:
        print(f"\nTEST FAILED: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
