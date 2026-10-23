"""Resume-to-job-role matching using TF-IDF + cosine similarity.

Compares the parsed resume text against a target job role. Produces a match
score (0-100), the list of matched required skills and the list of missing
required skills. Uses scikit-learn's TfidfVectorizer + cosine_similarity so
no external AI service is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .job_roles import get_required_skills


@dataclass
class ResumeMatchResult:
    """Result of comparing a resume against a target job role."""

    score: int  # 0-100
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    semantic_similarity: float = 0.0


def _semantic_match(resume_text: str, role: str, required_skills: list[str]) -> float:
    """Return cosine similarity (0-1) between resume text and a role profile.

    The role profile is built from the role name plus its required skills so the
    TF-IDF comparison has a meaningful document to compare against.
    """
    role_profile = f"{role} " + " ".join(required_skills)
    documents = [resume_text.lower(), role_profile.lower()]
    # Use character n-grams as well as word n-grams to be robust to short resumes
    # and partial skill names. sublinear_tf dampens very frequent terms.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        token_pattern=r"(?u)\b[\w+#.]+\b",
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Happens when both documents are empty after stop-word removal.
        return 0.0
    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
    return float(max(0.0, min(1.0, sim)))


def _keyword_skill_match(resume_text: str, required_skills: list[str]) -> tuple[list[str], list[str]]:
    """Split required skills into matched / missing based on resume mentions."""
    normalized_resume = resume_text.lower()
    matched: list[str] = []
    missing: list[str] = []
    for skill in required_skills:
        if skill in normalized_resume:
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def match_resume(resume_text: str, role: str) -> ResumeMatchResult:
    """Compute the resume match result for a given job role.

    The final score blends two signals:
      * semantic similarity between resume and role profile (TF-IDF cosine)
      * keyword coverage of the required skills list
    Both are scaled to 0-100 and combined 40/60 so that having the right
    keywords matters most, while overall topical similarity still contributes.
    """
    required_skills = get_required_skills(role)
    if not required_skills or not resume_text.strip():
        return ResumeMatchResult(score=0, matched_skills=[], missing_skills=required_skills)

    similarity = _semantic_match(resume_text, role, required_skills)
    matched, missing = _keyword_skill_match(resume_text, required_skills)

    coverage = len(matched) / len(required_skills) if required_skills else 0.0
    semantic_score = similarity * 100.0
    coverage_score = coverage * 100.0
    blended = 0.4 * semantic_score + 0.6 * coverage_score

    score = int(round(max(0.0, min(100.0, blended))))
    return ResumeMatchResult(
        score=score,
        matched_skills=matched,
        missing_skills=missing,
        semantic_similarity=round(similarity, 3),
    )
