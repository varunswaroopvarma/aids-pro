"""Skill gap analysis utilities.

Builds the existing / missing / suggested skill lists for a target job role
based on the skills extracted from the resume. Also produces short
forward-looking suggestions (technologies or concepts to learn next) derived
from the missing required skills plus a few role-specific stretch skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .job_roles import get_required_skills


# A small set of "stretch" skills per role to recommend beyond the required list.
# These give the suggested-skills section more depth.
STRETCH_SKILLS: dict[str, list[str]] = {
    "Java Developer": ["spring security", "reactive programming", "kubernetes", "aws"],
    "Python Developer": ["fastapi", "celery", "docker", "ci/cd"],
    "AI/ML Engineer": ["mlops", "model optimization", "hugging face", "langchain"],
    "Data Analyst": ["power bi", "sql window functions", "statistics", "python"],
    "Web Developer": ["typescript", "next js", "testing", "ci/cd"],
    "Software Engineer": ["system design", "distributed systems", "concurrency", "design patterns"],
}


@dataclass
class SkillGapResult:
    """Result of the skill gap analysis for a target role."""

    existing_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    suggested_skills: list[str] = field(default_factory=list)
    completeness: int = 0  # 0-100, fraction of required skills already present


def analyze_skill_gap(resume_skills: list[str], role: str) -> SkillGapResult:
    """Compare extracted resume skills with the required skills for a role."""
    required = get_required_skills(role)
    resume_set = {s.lower() for s in resume_skills}

    existing: list[str] = []
    missing: list[str] = []
    for skill in required:
        if skill.lower() in resume_set:
            existing.append(skill)
        else:
            missing.append(skill)

    completeness = int(round((len(existing) / len(required) * 100.0))) if required else 0

    # Suggested skills = missing required skills + a few stretch skills not
    # already present.
    suggested: list[str] = list(missing)
    for stretch in STRETCH_SKILLS.get(role, []):
        if stretch.lower() not in resume_set and stretch not in suggested:
            suggested.append(stretch)

    return SkillGapResult(
        existing_skills=existing,
        missing_skills=missing,
        suggested_skills=suggested,
        completeness=completeness,
    )
