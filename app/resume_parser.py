"""Resume parsing utilities.

Extracts raw text from an uploaded PDF resume using PyMuPDF and pulls out
candidate skills, education entries, project mentions and certifications using
lightweight keyword / pattern based heuristics. No external AI services are
used - everything runs locally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF


# Canonical skill vocabulary used to scan the resume text. Lower-cased, the
# matcher normalises the resume text the same way before searching.
SKILL_VOCABULARY = [
    # Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript", "go",
    "kotlin", "swift", "ruby", "php", "scala", "r", "dart", "rust",
    # Web / frontend
    "html", "css", "react", "angular", "vue", "next js", "node js", "express",
    "django", "flask", "fastapi", "spring", "spring boot", "asp.net",
    "tailwind", "bootstrap", "jquery", "webpack", "responsive design",
    # Data / ML
    "machine learning", "deep learning", "artificial intelligence", "nlp",
    "natural language processing", "computer vision", "tensorflow", "pytorch",
    "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "tableau", "power bi", "excel", "statistics", "data visualization",
    "data cleaning", "etl", "data wrangling", "hugging face", "transformers",
    "opencv", "feature engineering", "model deployment",
    # Backend / datastores
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite",
    "rest api", "graphql", "microservices", "docker", "kubernetes",
    "kafka", "celery", "hibernate", "jsp", "servlets", "jdbc",
    # Tools / practices
    "git", "github", "jenkins", "ci/cd", "aws", "azure", "gcp", "linux",
    "junit", "mockito", "pytest", "unittest", "testing",
    # Fundamentals
    "data structures", "algorithms", "operating systems", "dbms",
    "system design", "design patterns", "object oriented programming",
    "problem solving", "software engineering",
]

# Regex pattern for education degree keywords.
EDUCATION_PATTERNS = [
    r"\b(b\.?tech|bachelor of technology|bachelor of engineering|b\.?e\.?)\b",
    r"\b(m\.?tech|master of technology|master of engineering|m\.?e\.?)\b",
    r"\b(mba|master of business administration)\b",
    r"\b(mca|master of computer applications)\b",
    r"\b(bca|bachelor of computer applications)\b",
    r"\b(b\.?sc|bachelor of science)\b",
    r"\b(m\.?sc|master of science)\b",
    r"\b(ph\.?d|doctorate)\b",
    r"\b(diploma)\b",
]

# Headings that typically introduce a projects section in a resume.
PROJECT_SECTION_HEADERS = [
    "projects", "academic projects", "personal projects",
    "key projects", "project experience",
]

CERTIFICATION_KEYWORDS = [
    "certified", "certification", "certificate", "aws certified",
    "google certified", "microsoft certified", "coursera", "udemy",
    "npTEL", "nptel", "hackerrank", "leetcode certificate",
    "free code camp", "freecodecamp", "linkedin learning",
]


@dataclass
class ResumeData:
    """Structured result of parsing a resume PDF."""

    text: str = ""
    skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    line_count: int = 0


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Return the full plain-text content of a PDF using PyMuPDF.

    Raises ValueError when the PDF cannot be opened so the caller can surface
    a friendly error to the user.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - we want a single friendly error
        raise ValueError("Could not read the PDF file. Please upload a valid resume PDF.") from exc

    text_parts: list[str] = []
    try:
        for page in doc:
            text_parts.append(page.get_text("text"))
    finally:
        doc.close()

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise ValueError("The PDF appears to be empty or scanned. Please upload a text-based resume PDF.")
    return full_text


def _normalize(text: str) -> str:
    """Lower-case and collapse whitespace for keyword matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_skills(text: str) -> list[str]:
    """Scan resume text against the skill vocabulary and return matched skills."""
    normalized = _normalize(text)
    found: list[str] = []
    for skill in SKILL_VOCABULARY:
        # Use word-ish boundaries. For multi-word skills, simple substring is fine
        # because the vocabulary entries are distinctive enough.
        if " " in skill or "." in skill or "+" in skill or "#" in skill:
            if skill in normalized:
                found.append(skill)
        else:
            if re.search(rf"\b{re.escape(skill)}\b", normalized):
                found.append(skill)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for skill in found:
        if skill not in seen:
            seen.add(skill)
            unique.append(skill)
    return unique


def extract_education(text: str) -> list[str]:
    """Find degree keywords mentioned in the resume."""
    normalized = _normalize(text)
    found: list[str] = []
    for pattern in EDUCATION_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            label = match.group(1).upper()
            if label not in found:
                found.append(label)
    return found


def extract_projects(text: str) -> list[str]:
    """Return a short list of detected project bullet lines.

    We look for a projects section header and capture the next few non-empty
    lines, or fall back to lines that look like project bullets.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    projects: list[str] = []
    # Try locating a projects section header.
    start_idx = -1
    for idx, line in enumerate(lines):
        if any(header in line.lower() for header in PROJECT_SECTION_HEADERS):
            start_idx = idx
            break

    if start_idx >= 0:
        # Capture up to 6 following lines until the next all-caps section header.
        for line in lines[start_idx + 1 : start_idx + 10]:
            if len(line) < 3:
                continue
            if line.isupper() and len(line) < 30:
                break
            projects.append(line)
            if len(projects) >= 5:
                break

    # Fallback: lines that mention project/built/developed.
    if not projects:
        for line in lines:
            low = line.lower()
            if any(kw in low for kw in ("project", "built", "developed", "implemented")) and 10 < len(line) < 160:
                projects.append(line)
                if len(projects) >= 5:
                    break

    return projects


def extract_certifications(text: str) -> list[str]:
    """Find certification-related mentions in the resume."""
    normalized = _normalize(text)
    found: list[str] = []
    for kw in CERTIFICATION_KEYWORDS:
        if kw in normalized:
            # Keep the readable original casing form.
            found.append(kw.title() if kw.islower() else kw)
    # Deduplicate.
    seen: set[str] = set()
    unique: list[str] = []
    for cert in found:
        key = cert.lower()
        if key not in seen:
            seen.add(key)
            unique.append(cert)
    return unique


def parse_resume(pdf_bytes: bytes) -> ResumeData:
    """Parse a resume PDF and return structured ResumeData."""
    text = extract_text_from_pdf(pdf_bytes)
    return ResumeData(
        text=text,
        skills=extract_skills(text),
        education=extract_education(text),
        projects=extract_projects(text),
        certifications=extract_certifications(text),
        line_count=len([ln for ln in text.splitlines() if ln.strip()]),
    )
