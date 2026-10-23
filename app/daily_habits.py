"""Daily habits questionnaire definition and scoring logic.

Holds the 10 habit questions shown to the student, converts slider / dropdown
answers into a 0-100 Daily Habits Score using fixed, explainable weights, and
exposes the per-question contribution so the dashboard can show a breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HabitQuestion:
    """Definition of a single habit question."""

    key: str
    label: str
    help_text: str
    options: list[str]  # human-readable options, low -> high
    # Numeric value (0-1) assigned to each option index for scoring.
    weights: list[float]
    # Contribution weight (sums to 1.0 across all questions).
    contribution: float


# Each option maps to a 0..1 value. contribution values sum to 1.0.
HABIT_QUESTIONS: list[HabitQuestion] = [
    HabitQuestion(
        key="coding_hours",
        label="Coding Hours Per Day",
        help_text="Average number of hours you spend writing code daily.",
        options=["< 1 hour", "1-2 hours", "2-4 hours", "4-6 hours", "6+ hours"],
        weights=[0.1, 0.3, 0.55, 0.8, 1.0],
        contribution=0.15,
    ),
    HabitQuestion(
        key="dsa_practice",
        label="DSA Practice",
        help_text="How often you practice Data Structures & Algorithms problems.",
        options=["Rarely", "Weekly", "3-4 days/week", "Almost daily", "Daily + contests"],
        weights=[0.1, 0.3, 0.55, 0.8, 1.0],
        contribution=0.15,
    ),
    HabitQuestion(
        key="github_usage",
        label="GitHub Usage",
        help_text="Frequency of commits and contributions to GitHub.",
        options=["Never", "Occasionally", "Weekly", "Multiple times/week", "Daily"],
        weights=[0.05, 0.25, 0.5, 0.75, 1.0],
        contribution=0.12,
    ),
    HabitQuestion(
        key="learning_consistency",
        label="Learning Consistency",
        help_text="How consistently you study new topics over weeks.",
        options=["Inconsistent", "Somewhat", "Consistent", "Very consistent", "Highly disciplined"],
        weights=[0.1, 0.3, 0.55, 0.8, 1.0],
        contribution=0.10,
    ),
    HabitQuestion(
        key="mock_interviews",
        label="Mock Interviews",
        help_text="Participation in mock interviews or peer coding rounds.",
        options=["Never", "Rarely", "Monthly", "Weekly", "Multiple/week"],
        weights=[0.05, 0.2, 0.45, 0.7, 1.0],
        contribution=0.10,
    ),
    HabitQuestion(
        key="certifications",
        label="Certifications & Courses",
        help_text="Number of certifications or completed online courses.",
        options=["None", "1 course", "2-3 courses", "4-5 courses", "6+ / certified"],
        weights=[0.05, 0.3, 0.55, 0.8, 1.0],
        contribution=0.08,
    ),
    HabitQuestion(
        key="open_source",
        label="Open Source Contribution",
        help_text="Contributions to open source projects.",
        options=["None", "Exploring", "1 contribution", "A few PRs", "Regular contributor"],
        weights=[0.05, 0.2, 0.45, 0.7, 1.0],
        contribution=0.10,
    ),
    HabitQuestion(
        key="problem_solving",
        label="Problem Solving Confidence",
        help_text="Self-rated confidence tackling unseen problems.",
        options=["Low", "Below average", "Average", "Confident", "Very confident"],
        weights=[0.1, 0.3, 0.5, 0.75, 1.0],
        contribution=0.08,
    ),
    HabitQuestion(
        key="technical_reading",
        label="Technical Reading",
        help_text="How often you read blogs, papers or documentation.",
        options=["Rarely", "Occasionally", "Weekly", "Few times/week", "Daily"],
        weights=[0.1, 0.3, 0.5, 0.75, 1.0],
        contribution=0.07),
    HabitQuestion(
        key="new_tech",
        label="Learning New Technologies",
        help_text="How often you explore new frameworks or tools.",
        options=["Rarely", "Occasionally", "Monthly", "Weekly", "Continuously"],
        weights=[0.1, 0.3, 0.5, 0.75, 1.0],
        contribution=0.05,
    ),
]


@dataclass
class HabitScoreResult:
    """Result of scoring the daily habits questionnaire."""

    score: int  # 0-100
    answers: dict[str, str] = field(default_factory=dict)
    breakdown: list[dict] = field(default_factory=list)


def score_habits(answers: dict[str, int]) -> HabitScoreResult:
    """Compute the daily habits score from option-index answers.

    `answers` maps question key -> selected option index.
    """
    total = 0.0
    breakdown: list[dict] = []
    answer_labels: dict[str, str] = {}

    for question in HABIT_QUESTIONS:
        idx = answers.get(question.key, 0)
        idx = max(0, min(len(question.weights) - 1, int(idx)))
        value = question.weights[idx]
        contribution = value * question.contribution * 100.0
        total += contribution
        label = question.options[idx]
        answer_labels[question.key] = label
        breakdown.append({
            "question": question.label,
            "answer": label,
            "contribution": round(contribution, 1),
            "weight_percent": int(round(question.contribution * 100)),
        })

    score = int(round(max(0.0, min(100.0, total))))
    return HabitScoreResult(score=score, answers=answer_labels, breakdown=breakdown)
