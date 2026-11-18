
from agents.project_health_agent import ProjectHealthAgent


def test_scope_score():
    data = {
        "project_goals": ["Build an application"],
        "deliverables": ["Working application"],
        "milestones": ["M1"],
        "timelines": ["Week 1"],
        "responsibilities": ["Developer"],
        "missing_information": []
    }

    score = ProjectHealthAgent.calculate_scope_score(data)

    assert score == 100
    print("Scope score test passed:", score)


def test_timeline_score():
    data = {
        "delivery_forecast": {
            "status": "On Track"
        },
        "risks": []
    }

    score = ProjectHealthAgent.calculate_timeline_score(data)

    assert score == 100
    print("Timeline score test passed:", score)


def test_blocker_score():
    data = {
        "blockers": []
    }

    score = ProjectHealthAgent.calculate_blocker_score(data)

    assert score == 100
    print("Blocker score test passed:", score)


def test_overall_health_status():

    assert ProjectHealthAgent.get_health_status(90) == "Healthy"
    assert ProjectHealthAgent.get_health_status(70) == "Moderate"
    assert ProjectHealthAgent.get_health_status(50) == "At Risk"
    assert ProjectHealthAgent.get_health_status(30) == "Critical"

    print("Health status test passed")


if __name__ == "__main__":
    test_scope_score()
    test_timeline_score()
    test_blocker_score()
    test_overall_health_status()

    print("\nAll Project Health tests passed!")
