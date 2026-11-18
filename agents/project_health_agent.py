
from agents.scope_extraction_agent import ScopeExtractionAgent
from agents.risk_detection_agent import RiskDetectionAgent
from agents.blocker_action_agent import BlockerActionAgent


class ProjectHealthAgent:
    """Calculate project health using scope, timeline, and blockers."""

    def __init__(self):
        self.scope_agent = ScopeExtractionAgent()
        self.risk_agent = RiskDetectionAgent()
        self.blocker_agent = BlockerActionAgent()

    @staticmethod
    def calculate_scope_score(data):
        """Calculate scope clarity based on documented project details."""

        fields = [
            "project_goals",
            "deliverables",
            "milestones",
            "timelines",
            "responsibilities"
        ]

        completed = sum(
            1 for field in fields if data.get(field)
        )

        score = (completed / len(fields)) * 100

        missing = data.get("missing_information", [])
        score -= min(len(missing) * 5, 25)

        return max(0, round(score))

    @staticmethod
    def calculate_timeline_score(data):
        """Calculate timeline health using delivery forecast and risks."""

        forecast = data.get("delivery_forecast", {})
        status = str(forecast.get("status", "")).lower()

        if "on track" in status or "on_track" in status:
            score = 100
        elif "at risk" in status or "at_risk" in status:
            score = 60
        elif "delay" in status or "behind" in status:
            score = 30
        else:
            score = 50

        risks = data.get("risks", [])

        for risk in risks:
            likelihood = str(
                risk.get("likelihood", "unknown")
            ).lower()

            impact = str(
                risk.get("impact", "unknown")
            ).lower()

            if likelihood == "high" and impact == "high":
                score -= 15
            elif likelihood == "high" or impact == "high":
                score -= 10
            elif likelihood == "medium" and impact == "medium":
                score -= 5

        return max(0, min(100, round(score)))

    @staticmethod
    def calculate_blocker_score(data):
        """Calculate blocker health based on the number of blockers."""

        blockers = data.get("blockers", [])
        count = len(blockers)

        if count == 0:
            return 100
        elif count <= 2:
            return 80
        elif count <= 4:
            return 60
        elif count <= 6:
            return 40
        else:
            return 20

    @staticmethod
    def get_health_status(score):

        if score >= 80:
            return "Healthy"
        elif score >= 60:
            return "Moderate"
        elif score >= 40:
            return "At Risk"
        else:
            return "Critical"

    def calculate_project_health(self):

        # 1. Extract project scope
        scope_result = self.scope_agent.extract_scope(
            "Extract project goals, deliverables, milestones, "
            "timelines, responsibilities, and missing information."
        )

        # 2. Analyze project risks
        risk_result = self.risk_agent.detect_risks(
            "Identify project schedule risks, delivery challenges, "
            "risk likelihood, impact, and delivery forecast."
        )

        # 3. Identify project blockers
        blocker_result = self.blocker_agent.identify_blockers(
            "Identify all documented project blockers and unresolved dependencies."
        )

        results = [
            scope_result,
            risk_result,
            blocker_result
        ]

        if any(result.get("status") != "success" for result in results):
            return {
                "status": "incomplete",
                "message": (
                    "Unable to calculate complete project health. "
                    "Check whether project documents are indexed "
                    "and all three agents returned valid results."
                ),
                "scope_status": scope_result.get("status"),
                "risk_status": risk_result.get("status"),
                "blocker_status": blocker_result.get("status")
            }

        # Extract agent results
        scope_data = scope_result.get("extracted_data", {})
        risk_data = risk_result.get("risk_analysis", {})
        blocker_data = blocker_result.get("blocker_analysis", {})

        # Calculate individual scores
        scope_score = self.calculate_scope_score(scope_data)

        timeline_score = self.calculate_timeline_score(risk_data)

        blocker_score = self.calculate_blocker_score(blocker_data)

        # Weighted overall health score
        overall_score = round(
            (scope_score * 0.35) +
            (timeline_score * 0.35) +
            (blocker_score * 0.30)
        )

        return {
            "status": "success",
            "overall_score": overall_score,
            "health_status": self.get_health_status(overall_score),
            "dimension_scores": {
                "scope_clarity": scope_score,
                "timeline_health": timeline_score,
                "blocker_health": blocker_score
            },
            "blocker_count": len(
                blocker_data.get("blockers", [])
            ),
            "identified_risk_count": len(
                risk_data.get("risks", [])
            ),
            "delivery_forecast": risk_data.get(
                "delivery_forecast", {}
            ),
            "missing_information": scope_data.get(
                "missing_information", []
            )
        }
