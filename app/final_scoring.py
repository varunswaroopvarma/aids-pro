"""Final scoring: combines all sub-scores into the AI Job Readiness Index.

Updated weights reflecting the new employability dataset-based assessment:

    Employability (Rule-Based) ... 30%
    ML Prediction Confidence ..... 20%
    Resume Match ................. 25%
    GitHub Credibility ........... 15%
    Skill Completeness ........... 10%

The employability rule-based score comes directly from the 8 dataset features
(weighted average of 1-5 ratings normalised to 0-100).  The ML confidence is
the model's estimated probability for its predicted class — it measures model
certainty, not a guaranteed placement chance.  Resume match, GitHub
credibility and skill completeness remain unchanged from the original system.
"""

from __future__ import annotations

from dataclasses import dataclass


WEIGHTS = {
    "employability_rule": 0.30,
    "ml_confidence": 0.20,
    "resume_match": 0.25,
    "github_credibility": 0.15,
    "skill_completeness": 0.10,
}


@dataclass
class FinalScoreResult:
    """Final composite score and readiness level."""

    score: int
    level: str
    employability_rule: int
    ml_confidence: int
    resume_match: int
    github_credibility: int
    skill_completeness: int


def readiness_level(score: int) -> str:
    """Map a 0-100 score to a readiness level label."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Very Good"
    if score >= 55:
        return "Good"
    if score >= 40:
        return "Average"
    return "Needs Improvement"


def compute_final_score(
    employability_rule: int,
    ml_confidence: int,
    resume_match: int,
    github_credibility: int,
    skill_completeness: int,
) -> FinalScoreResult:
    """Compute the weighted AI Job Readiness Index.

    ml_confidence is the model's probability for its predicted class, scaled
    to 0-100 by the caller.
    """
    score = (
        employability_rule * WEIGHTS["employability_rule"]
        + ml_confidence * WEIGHTS["ml_confidence"]
        + resume_match * WEIGHTS["resume_match"]
        + github_credibility * WEIGHTS["github_credibility"]
        + skill_completeness * WEIGHTS["skill_completeness"]
    )
    score = int(round(max(0.0, min(100.0, score))))
    return FinalScoreResult(
        score=score,
        level=readiness_level(score),
        employability_rule=employability_rule,
        ml_confidence=ml_confidence,
        resume_match=resume_match,
        github_credibility=github_credibility,
        skill_completeness=skill_completeness,
    )