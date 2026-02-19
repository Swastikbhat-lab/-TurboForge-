"""
TurboForge - LLM Explanation Layer
Uses Claude API to generate natural language explanations of:
  - Failure predictions and causality
  - Counterfactual "what-if" scenarios
  - Operational recommendations
"""

import anthropic
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional


FEATURE_NAMES = [
    "wind_speed_ms",
    "rotor_rpm",
    "power_output_kw",
    "blade_pitch_deg",
    "nacelle_temp_c",
    "gearbox_temp_c",
    "generator_temp_c",
    "vibration_x",
    "vibration_y",
]

FEATURE_UNITS = {
    "wind_speed_ms": "m/s",
    "rotor_rpm": "RPM",
    "power_output_kw": "kW",
    "blade_pitch_deg": "°",
    "nacelle_temp_c": "°C",
    "gearbox_temp_c": "°C",
    "generator_temp_c": "°C",
    "vibration_x": "g",
    "vibration_y": "g",
}


@dataclass
class TurbineStatus:
    turbine_id: int
    failure_probability: float
    sensor_values: dict
    top_risk_features: list[str]
    counterfactual: Optional[dict] = None


class TurboForgeExplainer:
    """
    Natural language explainer for TurboForge predictions.
    Integrates with Claude API for human-readable diagnostics.
    """

    def __init__(self, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def _build_turbine_context(self, status: TurbineStatus) -> str:
        """Format turbine data into a structured context string."""
        sensors = "\n".join(
            f"  - {k}: {v:.3f} {FEATURE_UNITS.get(k, '')}"
            for k, v in status.sensor_values.items()
        )
        risk_features = ", ".join(status.top_risk_features)

        context = f"""
TURBINE STATUS REPORT
=====================
Turbine ID: {status.turbine_id}
Failure Probability (6h ahead): {status.failure_probability:.1%}
Risk Level: {"CRITICAL" if status.failure_probability > 0.7 else "HIGH" if status.failure_probability > 0.4 else "MODERATE" if status.failure_probability > 0.2 else "LOW"}

Current Sensor Readings:
{sensors}

Top Contributing Risk Factors: {risk_features}
"""
        if status.counterfactual:
            cf_str = "\n".join(
                f"  - {k}: {v:.3f} {FEATURE_UNITS.get(k, '')}"
                for k, v in status.counterfactual.items()
            )
            context += f"\nCounterfactual Scenario (What-If):\n{cf_str}"

        return context.strip()

    def explain_failure(self, status: TurbineStatus) -> str:
        """
        Generate a natural language failure explanation and recommendation.

        Args:
            status: TurbineStatus with sensor data and predictions

        Returns:
            Human-readable diagnostic report
        """
        context = self._build_turbine_context(status)

        prompt = f"""You are an expert wind turbine reliability engineer analyzing real-time SCADA data.

{context}

Please provide:
1. **Root Cause Analysis**: What is most likely causing the elevated failure risk? 
   Reference the specific sensor anomalies.
2. **Failure Mechanism**: Explain the physical mechanism that could lead to failure.
3. **Urgency Assessment**: How urgently should maintenance be dispatched?
4. **Recommended Actions**: Specific operational changes to reduce failure risk in the next 6 hours.
5. **Monitoring Priorities**: Which sensors to watch most closely.

Keep your response concise, technical, and actionable. Use bullet points where appropriate."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def explain_counterfactual(
        self,
        status: TurbineStatus,
        original_prob: float,
        counterfactual_prob: float,
        changed_feature: str,
        changed_value: float,
    ) -> str:
        """
        Explain the impact of a what-if scenario on failure probability.
        """
        direction = "decreased" if counterfactual_prob < original_prob else "increased"
        delta = abs(counterfactual_prob - original_prob)

        prompt = f"""You are a wind energy systems analyst reviewing a counterfactual scenario.

ORIGINAL SCENARIO:
- Turbine {status.turbine_id} failure probability: {original_prob:.1%}

WHAT-IF SCENARIO:
- If {changed_feature} were changed to {changed_value:.3f} {FEATURE_UNITS.get(changed_feature, '')}
- New failure probability: {counterfactual_prob:.1%}
- Change: {direction} by {delta:.1%}

Explain in 3-4 sentences:
1. Why this specific change {direction} the failure risk
2. The physical/mechanical reasoning
3. Whether this intervention is practically feasible as an operational control action"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def fleet_summary(self, turbine_statuses: list[TurbineStatus]) -> str:
        """
        Generate a fleet-level operational summary for all 50 turbines.
        """
        critical = [s for s in turbine_statuses if s.failure_probability > 0.7]
        high = [s for s in turbine_statuses if 0.4 < s.failure_probability <= 0.7]
        moderate = [s for s in turbine_statuses if 0.2 < s.failure_probability <= 0.4]
        low = [s for s in turbine_statuses if s.failure_probability <= 0.2]

        avg_prob = np.mean([s.failure_probability for s in turbine_statuses])

        fleet_data = f"""
FLEET SUMMARY — {len(turbine_statuses)} Turbines
==============================
Average Fleet Failure Probability: {avg_prob:.1%}

Risk Distribution:
  🔴 CRITICAL (>70%): {len(critical)} turbines — IDs: {[s.turbine_id for s in critical]}
  🟠 HIGH (40-70%):   {len(high)} turbines — IDs: {[s.turbine_id for s in high]}
  🟡 MODERATE (20-40%): {len(moderate)} turbines
  🟢 LOW (<20%):      {len(low)} turbines

Top Failing Turbines:
"""
        top5 = sorted(turbine_statuses, key=lambda s: s.failure_probability, reverse=True)[:5]
        for s in top5:
            fleet_data += f"  - Turbine {s.turbine_id}: {s.failure_probability:.1%} | Risk: {', '.join(s.top_risk_features[:2])}\n"

        prompt = f"""You are an operations manager for a 50-turbine wind farm. Here is the current status:

{fleet_data.strip()}

Provide a concise executive briefing (5-7 sentences) covering:
1. Overall fleet health assessment
2. Immediate maintenance priorities
3. Expected impact on power production if failures occur
4. Recommended operational strategy for the next 6 hours"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# ─────────────────────────────────────────────
# Demo Usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    explainer = TurboForgeExplainer()

    # Simulate a high-risk turbine
    status = TurbineStatus(
        turbine_id=23,
        failure_probability=0.82,
        sensor_values={
            "wind_speed_ms": 11.2,
            "rotor_rpm": 14.5,
            "power_output_kw": 1640.0,
            "blade_pitch_deg": 6.3,
            "nacelle_temp_c": 48.2,
            "gearbox_temp_c": 71.8,   # Elevated
            "generator_temp_c": 88.4,  # High
            "vibration_x": 0.068,      # 3x normal
            "vibration_y": 0.051,
        },
        top_risk_features=["gearbox_temp_c", "vibration_x", "generator_temp_c"],
    )

    print("=" * 60)
    print("FAILURE EXPLANATION")
    print("=" * 60)
    explanation = explainer.explain_failure(status)
    print(explanation)

    print("\n" + "=" * 60)
    print("COUNTERFACTUAL ANALYSIS")
    print("=" * 60)
    cf_explanation = explainer.explain_counterfactual(
        status=status,
        original_prob=0.82,
        counterfactual_prob=0.34,
        changed_feature="blade_pitch_deg",
        changed_value=12.0,
    )
    print(cf_explanation)
