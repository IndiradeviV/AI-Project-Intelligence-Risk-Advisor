
from agents.project_intelligence_agent import ProjectIntelligenceAgent


def test_assistant_accuracy():

    agent = ProjectIntelligenceAgent()

    test_cases = [
        {
            "question": "Who is responsible for implementing registration and login APIs?",
            "expected_keywords": ["Ravi"]
        },
        {
            "question": "Who is responsible for building login and registration screens?",
            "expected_keywords": ["Meera"]
        },
        {
            "question": "What are the main project deliverables?",
            "expected_keywords": [
                "proposal",
                "requirements",
                "frontend",
                "backend"
            ]
        },
        {
            "question": "What are the biggest project risks?",
            "expected_keywords": [
                "integration",
                "adoption",
                "testing"
            ]
        }
    ]

    passed = 0
    failed = 0

    for index, test in enumerate(test_cases, start=1):

        print("\n" + "=" * 60)
        print(f"TEST CASE {index}")
        print("Question:", test["question"])

        result = agent.ask_question(test["question"])

        if result.get("status") != "success":
            print("FAILED: Assistant could not generate an answer.")
            failed += 1
            continue

        answer = result.get("answer", "").lower()

        print("\nGenerated Answer:")
        print(result["answer"])

        matched = [
            keyword
            for keyword in test["expected_keywords"]
            if keyword.lower() in answer
        ]

        print("\nMatched keywords:", matched)

        if matched:
            print("RESULT: PASSED")
            passed += 1
        else:
            print("RESULT: FAILED")
            failed += 1

    print("\n" + "=" * 60)
    print("ACCURACY VALIDATION SUMMARY")
    print("=" * 60)
    print("Total test cases:", len(test_cases))
    print("Passed:", passed)
    print("Failed:", failed)

    if failed == 0:
        print("All validation tests passed!")
    else:
        print("Some validation tests need improvement.")


if __name__ == "__main__":
    test_assistant_accuracy()
