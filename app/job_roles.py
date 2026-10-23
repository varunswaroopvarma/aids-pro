"""Job role definitions and required skills mapping.

Central registry of the target job roles supported by the app along with the
keyword skills used for resume matching and skill-gap analysis. Keeping the
data here makes it easy to extend the role list without touching the scoring
logic.
"""

# Canonical list of job roles shown in the dropdown (order matters for UI).
JOB_ROLES = [
    "Java Developer",
    "Python Developer",
    "AI/ML Engineer",
    "Data Analyst",
    "Web Developer",
    "Software Engineer",
]

# Required skills per job role. Each entry holds a list of canonical skill
# keywords. The matching engine normalises resume text against these.
JOB_ROLE_SKILLS = {
    "Java Developer": [
        "java", "spring", "spring boot", "hibernate", "jsp", "servlets",
        "maven", "gradle", "rest api", "jdbc", "kafka", "junit", "mockito",
        "sql", "microservices", "docker", "kubernetes", "git",
    ],
    "Python Developer": [
        "python", "django", "flask", "fastapi", "pandas", "numpy",
        "rest api", "sql", "postgresql", "redis", "celery", "docker",
        "git", "unittest", "pytest", "api", "linux",
    ],
    "AI/ML Engineer": [
        "python", "machine learning", "deep learning", "tensorflow",
        "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "natural language processing", "nlp", "computer vision",
        "transformers", "hugging face", "opencv", "matplotlib",
        "feature engineering", "model deployment",
    ],
    "Data Analyst": [
        "sql", "excel", "power bi", "tableau", "python", "pandas",
        "numpy", "matplotlib", "seaborn", "statistics", "data cleaning",
        "data visualization", "etl", "power query", "r", "data wrangling",
    ],
    "Web Developer": [
        "html", "css", "javascript", "react", "angular", "vue",
        "node js", "express", "mongodb", "sql", "rest api", "typescript",
        "tailwind", "bootstrap", "git", "responsive design", "webpack",
    ],
    "Software Engineer": [
        "data structures", "algorithms", "java", "python", "c++",
        "operating systems", "dbms", "sql", "git", "system design",
        "design patterns", "rest api", "docker", "linux", "testing",
        "object oriented programming", "problem solving",
    ],
}

# Friendly descriptions shown on the Home / Student Details screens.
JOB_ROLE_DESCRIPTIONS = {
    "Java Developer": "Builds robust backend services and enterprise apps using Java ecosystems.",
    "Python Developer": "Develops web services, automation and data pipelines using Python.",
    "AI/ML Engineer": "Designs and deploys machine learning and deep learning models.",
    "Data Analyst": "Turns raw data into insights using SQL, BI tools and statistics.",
    "Web Developer": "Builds responsive, interactive web applications end to end.",
    "Software Engineer": "Strong fundamentals in DSA, system design and multi-language development.",
}


def get_required_skills(role: str) -> list[str]:
    """Return the canonical required skills list for a job role."""
    return list(JOB_ROLE_SKILLS.get(role, []))


def get_role_description(role: str) -> str:
    """Return a short human-readable description of a job role."""
    return JOB_ROLE_DESCRIPTIONS.get(role, "")
