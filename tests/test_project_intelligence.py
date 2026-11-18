
from agents.project_intelligence_agent import ProjectIntelligenceAgent


def test_project_intelligence():

    agent = ProjectIntelligenceAgent()

    questions = [
        "What are the main project deliverables?",
        "Who is responsible for implementing registration and login?",
        "What are the biggest project risks?",
        "Are there any documented project blockers?"
    ]

    for index, question in enumerate(questions, start=1):

        print("\n" + "=" * 60)
        print(f"TEST {index}: {question}")
        print("=" * 60)

        result = agent.ask_question(question)

        if result.get("status") == "success":

            print("\nAnswer:")
            print(result.get("answer"))

            print("\nSource chunks:")
            print(result.get("source_chunks"))

            print("\nSource documents:")

            for source in result.get("sources", []):
                print(
                    f"- {source['filename']} "
                    f"(Chunk {source['chunk']})"
                )

        else:
            print("\nStatus:", result.get("status"))
            print("Message:", result.get("message"))

    print("\nConversational Agent testing completed.")


if __name__ == "__main__":
    test_project_intelligence()
