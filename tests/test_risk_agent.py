import json

from agents.risk_detection_agent import RiskDetectionAgent


def main():
    print("=" * 60)
    print("RISK DETECTION AGENT - TEST")
    print("=" * 60)

    try:
        agent = RiskDetectionAgent()

        query = (
            "Identify project risks, schedule risks, dependency gaps, "
            "delivery challenges, and possible mitigation actions. "
            "Assess delivery status only from the available evidence."
        )

        print("\nAnalyzing indexed project documents...\n")

        result = agent.detect_risks(query)

        print("TEST RESULT:")
        print(json.dumps(result, indent=4, ensure_ascii=False))

        if result.get("status") == "success":
            print("\nSUCCESS: Risk analysis completed.")
        elif result.get("status") == "no_context":
            print("\nNO CONTEXT: Upload and index project documents first.")
        else:
            print("\nCHECK REQUIRED: The agent did not return valid JSON.")

    except Exception as error:
        print(f"\nTEST FAILED: {type(error).__name__}: {error}")
        print("Check the error message before making any code changes.")


if __name__ == "__main__":
    main()
