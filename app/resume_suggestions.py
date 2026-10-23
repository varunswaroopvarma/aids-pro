"""Rule-based resume improvement suggestions.

Generates concrete, actionable suggestions based on the parsed resume content
and the match / skill-gap results. No AI services are used - every suggestion
comes from a deterministic rule.
"""

from __future__ import annotations

from .resume_parser import ResumeData
from .resume_matcher import ResumeMatchResult
from .skill_gap import SkillGapResult


def generate_suggestions(
    resume: ResumeData,
    match: ResumeMatchResult,
    skill_gap: SkillGapResult,
    github_readme_present: bool = False,
) -> list[str]:
    """Return a list of tailored resume improvement suggestions."""
    suggestions: list[str] = []

    # --- Skills coverage ---
    if match.score < 50:
        suggestions.append(
            "Your resume mentions fewer than half of the key skills for this role. "
            "Add a dedicated 'Skills' section listing relevant technologies."
        )
    elif match.score < 75:
        suggestions.append(
            "You have a moderate skill match. Highlight the strongest relevant skills "
            "near the top of your resume."
        )

    if skill_gap.missing_skills:
        top_missing = ", ".join(skill_gap.missing_skills[:4])
        suggestions.append(
            f"Missing key skills for this role: {top_missing}. "
            "If you have any exposure to them, mention it; otherwise prioritise learning them."
        )

    # --- Projects ---
    if len(resume.projects) == 0:
        suggestions.append(
            "No clear projects section detected. Add 2-3 projects with the problem, "
            "your role, the tech stack and a measurable outcome."
        )
    elif len(resume.projects) < 2:
        suggestions.append(
            "Only one project detected. Add at least two more projects to demonstrate breadth."
        )
    else:
        suggestions.append(
            "Strengthen project descriptions: start each bullet with an action verb and "
            "add a measurable achievement (e.g. 'reduced load time by 30%')."
        )

    # --- Certifications ---
    if len(resume.certifications) == 0:
        suggestions.append(
            "No certifications detected. Add relevant certifications or completed courses "
            "to strengthen credibility."
        )

    # --- Education ---
    if len(resume.education) == 0:
        suggestions.append(
            "No education details detected. Add your degree, institution and graduation year."
        )

    # --- Resume length ---
    if resume.line_count < 20:
        suggestions.append(
            "Your resume seems short. Expand experience, projects and skills to at least one full page."
        )
    elif resume.line_count > 90:
        suggestions.append(
            "Your resume is quite long. Trim older or irrelevant content to keep it to 1-2 pages."
        )

    # --- GitHub / README ---
    if not github_readme_present:
        suggestions.append(
            "Your GitHub repository has no README. Add a clear README with project overview, "
            "setup steps and screenshots."
        )

    # --- Measurable achievements heuristic ---
    text_lower = resume.text.lower()
    if not any(kw in text_lower for kw in ["%", "increased", "reduced", "improved", "achieved", "built"]):
        suggestions.append(
            "Add measurable achievements - numbers and impact make your experience stand out."
        )

    # Always include one positive, forward-looking note.
    if not suggestions:
        suggestions.append(
            "Your resume looks well aligned with the role. Keep it updated with recent projects."
        )

    return suggestions
