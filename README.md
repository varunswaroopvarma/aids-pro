---
title: Students Ranking System
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.60.0"
app_file: app.py
pinned: false
license: mit
---

# Students' Ranking System Based on Daily Habits Estimation for Job Selection using AI Techniques

A modern, lightweight Streamlit web application that evaluates a student's job
readiness using **resume analysis**, **daily habits assessment**, **GitHub
project credibility analysis** and **skill gap analysis**.

The app runs entirely on CPU and uses **no LLM APIs or paid AI services** -
every score is computed with explainable, rule-based logic.

---

## Features

- **Resume Analysis** - upload a PDF resume; skills, education, projects and
  certifications are extracted with PyMuPDF and keyword heuristics.
- **Resume Match** - TF-IDF + cosine similarity compares the resume against the
  selected job role and reports matched / missing skills.
- **Daily Habits Assessment** - a 10-question questionnaire produces a 0-100
  habits score with a per-question contribution breakdown.
- **GitHub Project Credibility** - repository metadata (age, activity, stars,
  forks, README, languages, topics) is scored with fixed weights and clear
  reasons.
- **Skill Gap Analysis** - existing, missing and suggested skills for the
  target role.
- **AI Job Readiness Index** - a weighted 0-100 index with readiness levels
  (Excellent / Very Good / Good / Average / Needs Improvement).
- **PDF Report** - downloadable evaluation report built with ReportLab.
- **Session Leaderboard** - compare students during the current session
  (in-memory only, no database).

## Scoring Weights

| Component            | Weight |
|----------------------|--------|
| Resume Match         | 45%    |
| Daily Habits         | 25%    |
| GitHub Credibility   | 20%    |
| Skill Completeness   | 10%    |

## Technology Stack

- **Frontend / app:** Streamlit
- **Language:** Python 3.11
- **PDF parsing:** PyMuPDF (`fitz`)
- **Resume matching:** scikit-learn (TF-IDF + cosine similarity)
- **GitHub analysis:** public GitHub REST API (`requests`)
- **Charts:** Plotly
- **PDF report:** ReportLab
- **Deployment:** Hugging Face Spaces (CPU)

## Project Structure

```
.
├── app.py                  # Main Streamlit app (all screens + routing)
├── requirements.txt
├── README.md
└── app/                    # Application package
    ├── __init__.py
    ├── job_roles.py        # Job role -> required skills registry
    ├── resume_parser.py    # PDF text extraction + section heuristics
    ├── resume_matcher.py   # TF-IDF cosine similarity matching
    ├── github_analyzer.py  # GitHub REST API credibility scoring
    ├── daily_habits.py     # 10-question habits questionnaire + scoring
    ├── skill_gap.py        # Existing / missing / suggested skills
    ├── final_scoring.py    # Weighted AI Job Readiness Index
    ├── resume_suggestions.py  # Rule-based improvement suggestions
    ├── pdf_report.py       # ReportLab PDF generation
    ├── leaderboard.py      # In-memory session leaderboard
    └── charts.py           # Plotly gauge / radar / progress charts
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Deployment on Hugging Face Spaces (CPU)

1. Create a new **Space** and choose **SDK: Streamlit**.
2. Upload all files (keep the `app/` folder and `app.py` at the repository root).
3. Ensure `requirements.txt` is present - the Space installs dependencies
   automatically.
4. The Space runs on the free **CPU** tier. No GPU, no database and no paid
   APIs are required.

## Important Constraints

- **No LLM APIs** (OpenAI, Gemini, Claude, Groq, Hugging Face Inference, etc.).
- The only network call is to the **public GitHub REST API**.
- Resume matching uses **TF-IDF + cosine similarity**.
- Skill gap analysis uses **keyword matching**.
- Resume suggestions use **rule-based logic**.
- The GitHub credibility score is computed from **repository metadata**.
- The leaderboard is **session-only** (in-memory, no persistence).

## Job Roles Supported

Java Developer, Python Developer, AI/ML Engineer, Data Analyst,
Web Developer, Software Engineer.

## License

This project is intended for educational use as a final-year engineering
project.
