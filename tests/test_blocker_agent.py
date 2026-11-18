
import json

from agents.blocker_action_agent import BlockerActionAgent


def main():
    print("=" * 60)
    print("BLOCKER AND ACTION ITEM AGENT - TEST")
    print("=" * 60)

    try:
        agent = BlockerActionAgent()

        query = (
            "Identify documented project blockers, action items, "
            "unresolved dependencies, responsible owners, deadlines, "
            "and missing information."
        )

        print("\nAnalyzing project documents...\n")

        result = agent.identify_blockers(query)

        print("TEST RESULT:")
        print(json.dumps(result, indent=4, ensure_ascii=False))

        if result.get("status") == "success":
            print("\nSUCCESS: Blocker analysis completed.")
        elif result.get("status") == "no_context":
            print("\nNO CONTEXT: No project documents were retrieved.")
        else:
            print("\nCHECK REQUIRED: The agent returned invalid JSON.")

    except Exception as error:
        print(f"\nTEST FAILED: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
