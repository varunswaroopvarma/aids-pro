"""Employability assessment form based on the Student-Employability dataset.

Defines 8 rating fields matching the actual feature columns of the uploaded
CSV.  Each field uses a 1-5 integer scale identical to the dataset.  A
rule-based score (0-100) is computed from the same inputs using clearly
documented weights, and the form data is passed to the ML model for a
separate prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmployabilityField:
    """A single rating field on the employability form."""

    key: str
    label: str
    help_text: str
    weight: float


EMPLOYABILITY_FIELDS: list[EmployabilityField] = [
    EmployabilityField(
        key="GENERAL APPEARANCE",
        label="General Appearance",
        help_text="Professional dress, grooming and overall presentation.",
        weight=0.12,
    ),
    EmployabilityField(
        key="MANNER OF SPEAKING",
        label="Manner of Speaking",
        help_text="Clarity, tone and articulation during conversation.",
        weight=0.14,
    ),
    EmployabilityField(
        key="PHYSICAL CONDITION",
        label="Physical Condition",
        help_text="Energy level, posture and physical presence.",
        weight=0.08,
    ),
    EmployabilityField(
        key="MENTAL ALERTNESS",
        label="Mental Alertness",
        help_text="Quickness of thought, attentiveness and responsiveness.",
        weight=0.14,
    ),
    EmployabilityField(
        key="SELF-CONFIDENCE",
        label="Self-Confidence",
        help_text="Confidence in expressing ideas and handling questions.",
        weight=0.12,
    ),
    EmployabilityField(
        key="ABILITY TO PRESENT IDEAS",
        label="Ability to Present Ideas",
        help_text="How well ideas are structured and communicated.",
        weight=0.14,
    ),
    EmployabilityField(
        key="COMMUNICATION SKILLS",
        label="Communication Skills",
        help_text="Listening, verbal expression and interpersonal clarity.",
        weight=0.14,
    ),
    EmployabilityField(
        key="Student Performance Rating",
        label="Student Performance Rating",
        help_text="Overall academic / practical performance rating.",
        weight=0.12,
    ),
]

assert abs(sum(f.weight for f in EMPLOYABILITY_FIELDS) - 1.0) < 1e-9

RATING_MIN = 1
RATING_MAX = 5


@dataclass
class EmployabilityRuleResult:
    """Result of the rule-based employability scoring."""

    score: int
    answers: dict[str, int] = field(default_factory=dict)
    breakdown: list[dict] = field(default_factory=list)


@dataclass
class MLPredictionResult:
    """Result of the ML model prediction."""

    predicted_class: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


def score_employability_rule(answers: dict[str, int]) -> EmployabilityRuleResult:
    """Compute a rule-based employability score (0-100) from 1-5 ratings.

    Each field's rating is normalised to 0-1 via (rating - 1) / 4, then
    multiplied by its weight and scaled to 100.
    """
    total = 0.0
    breakdown: list[dict] = []

    for field_def in EMPLOYABILITY_FIELDS:
        raw = answers.get(field_def.key, RATING_MIN)
        raw = max(RATING_MIN, min(RATING_MAX, int(raw)))
        normalised = (raw - RATING_MIN) / (RATING_MAX - RATING_MIN)
        contribution = normalised * field_def.weight * 100.0
        total += contribution
        breakdown.append({
            "field": field_def.label,
            "rating": raw,
            "contribution": round(contribution, 1),
            "weight_percent": int(round(field_def.weight * 100)),
        })

    score = int(round(max(0.0, min(100.0, total))))
    return EmployabilityRuleResult(
        score=score,
        answers={k: v for k, v in answers.items()},
        breakdown=breakdown,
    )