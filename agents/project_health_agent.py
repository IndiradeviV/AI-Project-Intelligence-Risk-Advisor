
class ProjectHealthAgent:
    """Calculate project health using existing agent results."""

    @staticmethod
    def calculate_scope_score(data):
        fields = [
            "project_goals",
            "deliverables",
            "milestones",
            "timelines",
            "responsibilities"
        ]

        completed = sum(1 for field in fields if data.get(field))

        score = (completed / len(fields)) * 100

        missing = data.get("missing_information", [])
        score -= min(len(missing) * 5, 25)

        return max(0, round(score))

    @staticmethod
    def calculate_timeline_score(data):
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

    def calculate_project_health(
        self,
        scope_result,
        risk_result,
        blocker_result
    ):
        """Calculate health from already-generated agent results."""

        results = [
            scope_result,
            risk_result,
            blocker_result
        ]

        if any(
            not isinstance(result, dict)
            or result.get("status") != "success"
            for result in results
        ):
            return {
                "status": "incomplete",
                "message": (
                    "Run Scope Analysis, Risk Analysis and "
                    "Blocker Analysis successfully before "
                    "calculating project health."
                ),
                "scope_status": (
                    scope_result.get("status")
                    if isinstance(scope_result, dict)
                    else "missing"
                ),
                "risk_status": (
                    risk_result.get("status")
                    if isinstance(risk_result, dict)
                    else "missing"
                ),
                "blocker_status": (
                    blocker_result.get("status")
                    if isinstance(blocker_result, dict)
                    else "missing"
                )
            }

        scope_data = scope_result.get("extracted_data", {})
        risk_data = risk_result.get("risk_analysis", {})
        blocker_data = blocker_result.get("blocker_analysis", {})

        scope_score = self.calculate_scope_score(scope_data)
        timeline_score = self.calculate_timeline_score(risk_data)
        blocker_score = self.calculate_blocker_score(blocker_data)

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
