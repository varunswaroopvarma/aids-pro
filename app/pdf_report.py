"""PDF report generation using ReportLab.

Updated to include the employability assessment (rule-based score + ML
prediction) instead of the old daily habits questionnaire.
"""

from __future__ import annotations

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

from .final_scoring import FinalScoreResult
from .resume_matcher import ResumeMatchResult
from .skill_gap import SkillGapResult
from .github_analyzer import GitHubCredibilityResult
from .employability_form import EmployabilityRuleResult, MLPredictionResult
from .resume_suggestions import generate_suggestions
from .resume_parser import ResumeData


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleX", parent=base["Title"], fontSize=20, textColor=colors.HexColor("#0f172a"), spaceAfter=4),
        "subtitle": ParagraphStyle("SubtitleX", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#475569"), spaceAfter=10),
        "h2": ParagraphStyle("H2X", parent=base["Heading2"], fontSize=13, textColor=colors.HexColor("#1d4ed8"), spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("BodyX", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#1f2937"), leading=14),
        "small": ParagraphStyle("SmallX", parent=base["Normal"], fontSize=9, textColor=colors.HexColor("#475569")),
        "bullet": ParagraphStyle("BulletX", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#1f2937"), leftIndent=12, bulletIndent=2, leading=14),
        "scorebig": ParagraphStyle("ScoreBig", parent=base["Title"], fontSize=28, textColor=colors.HexColor("#1d4ed8"), alignment=1, spaceAfter=2),
        "scorelabel": ParagraphStyle("ScoreLabel", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#475569"), alignment=1),
    }


def _score_table(scores: dict[str, tuple[int, float]]) -> Table:
    data = [["Metric", "Score (0-100)", "Weight"]]
    for label, (score, weight) in scores.items():
        data.append([label, str(score), f"{int(weight * 100)}%"])
    table = Table(data, colWidths=[80 * mm, 40 * mm, 30 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _bullets(items: list[str], style) -> list:
    return [Paragraph(f"&bull; {item}", style) for item in items]


def generate_pdf_report(
    student_name: str,
    job_role: str,
    resume: ResumeData,
    match: ResumeMatchResult,
    employability: EmployabilityRuleResult,
    ml_prediction: MLPredictionResult,
    github: GitHubCredibilityResult,
    skill_gap: SkillGapResult,
    final: FinalScoreResult,
) -> bytes:
    """Generate the evaluation PDF report and return it as bytes."""
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Job Readiness Report", author="Students Ranking System",
    )
    styles = _styles()
    story: list = []

    story.append(Paragraph("Job Readiness Report", styles["title"]))
    story.append(Paragraph("Students' Ranking System — Employability Assessment", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Student Details", styles["h2"]))
    details = [
        ["Name", student_name],
        ["Target Job Role", job_role],
        ["Date Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["GitHub Repository", github.repo_url or "Not provided"],
    ]
    details_table = Table(details, colWidths=[40 * mm, 120 * mm])
    details_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1d4ed8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(details_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("AI Job Readiness Index", styles["h2"]))
    story.append(Paragraph(str(final.score), styles["scorebig"]))
    story.append(Paragraph(final.level, styles["scorelabel"]))
    story.append(Spacer(1, 6))

    scores = {
        "Employability (Rule-Based)": (final.employability_rule, 0.30),
        "ML Prediction Confidence": (final.ml_confidence, 0.20),
        "Resume Match": (final.resume_match, 0.25),
        "GitHub Credibility": (final.github_credibility, 0.15),
        "Skill Completeness": (final.skill_completeness, 0.10),
    }
    story.append(_score_table(scores))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Resume Match Analysis", styles["h2"]))
    story.append(Paragraph(f"Resume Match Score: <b>{match.score}/100</b>", styles["body"]))
    if match.matched_skills:
        story.append(Paragraph("Matching skills: " + ", ".join(match.matched_skills), styles["body"]))
    if match.missing_skills:
        story.append(Paragraph("Missing skills: " + ", ".join(match.missing_skills), styles["body"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Extracted Resume Information", styles["h2"]))
    story.append(Paragraph("Skills: " + (", ".join(resume.skills) if resume.skills else "None detected"), styles["body"]))
    story.append(Paragraph("Education: " + (", ".join(resume.education) if resume.education else "None detected"), styles["body"]))
    story.append(Paragraph("Certifications: " + (", ".join(resume.certifications) if resume.certifications else "None detected"), styles["body"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Employability Assessment", styles["h2"]))
    story.append(Paragraph(f"Rule-Based Employability Score: <b>{employability.score}/100</b>", styles["body"]))
    for item in employability.breakdown:
        story.append(Paragraph(f"&bull; {item['field']}: Rating {item['rating']}/5 (+{item['contribution']})", styles["bullet"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"ML Prediction: <b>{ml_prediction.predicted_class}</b>", styles["body"]))
    prob_text = ", ".join(f"{cls}: {p*100:.1f}%" for cls, p in ml_prediction.probabilities.items())
    story.append(Paragraph(f"Model-estimated probabilities: {prob_text}", styles["body"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("GitHub Project Credibility", styles["h2"]))
    story.append(Paragraph(f"Credibility Score: <b>{github.score}/100</b>", styles["body"]))
    if github.repo_name:
        story.append(Paragraph(f"Repository: {github.repo_name}", styles["body"]))
    for reason in github.reasons:
        story.append(Paragraph(f"&bull; {reason}", styles["bullet"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Skill Gap Analysis", styles["h2"]))
    story.append(Paragraph("Existing skills: " + (", ".join(skill_gap.existing_skills) if skill_gap.existing_skills else "None"), styles["body"]))
    story.append(Paragraph("Missing skills: " + (", ".join(skill_gap.missing_skills) if skill_gap.missing_skills else "None"), styles["body"]))
    story.append(Paragraph("Suggested skills to learn: " + (", ".join(skill_gap.suggested_skills) if skill_gap.suggested_skills else "None"), styles["body"]))
    story.append(Spacer(1, 6))

    suggestions = generate_suggestions(resume, match, skill_gap, github.has_readme)
    story.append(Paragraph("Resume Improvement Suggestions", styles["h2"]))
    for s in suggestions:
        story.append(Paragraph(f"&bull; {s}", styles["bullet"]))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1")))
    story.append(Paragraph(
        "This report was generated by the Students' Ranking System. The employability "
        "score uses a Random Forest model trained on the Student-Employability dataset "
        "with 5-fold cross-validation. Scores are computed using rule-based logic, "
        "TF-IDF resume matching and GitHub repository metadata.",
        styles["small"],
    ))

    doc.build(story)
    return buffer.getvalue()