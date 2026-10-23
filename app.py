"""Main Streamlit application for the Students' Ranking System.

Updated to use the employability dataset form (8 rating fields) instead of the
old daily habits questionnaire.  Both a rule-based score and an ML-based
prediction are computed from the same form inputs.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as _dt

import streamlit as st

from app import (
    employability_form,
    final_scoring,
    github_analyzer,
    job_roles,
    leaderboard,
    ml_model,
    pdf_report,
    resume_matcher,
    resume_parser,
    resume_suggestions,
    skill_gap,
)
from app import charts

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Students' Ranking System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        --primary: #1d4ed8;
        --primary-light: #3b82f6;
        --slate-900: #0f172a;
        --slate-700: #334155;
        --slate-500: #64748b;
        --slate-200: #e2e8f0;
        --slate-50: #f8fafc;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    h1, h2, h3 {
        color: var(--slate-900) !important;
        letter-spacing: -0.02em;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid var(--slate-200);
        border-radius: 14px;
        padding: 1.4rem 1.3rem;
        height: 100%;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        border-color: var(--primary-light);
    }
    .feature-card .icon {
        font-size: 1.6rem;
        margin-bottom: .5rem;
        display: block;
    }
    .feature-card h3 {
        font-size: 1.05rem;
        margin: 0 0 .4rem 0;
        color: var(--slate-900);
    }
    .feature-card p {
        font-size: .9rem;
        color: var(--slate-500);
        margin: 0;
        line-height: 1.5;
    }
    .score-card {
        background: white;
        border: 1px solid var(--slate-200);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
    }
    .score-card .value {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-card .label {
        font-size: .85rem;
        color: var(--slate-500);
        margin-top: .35rem;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .hero-badge {
        display: inline-block;
        background: #dbeafe;
        color: var(--primary);
        font-size: .8rem;
        font-weight: 600;
        padding: .35rem .8rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .stat-pill {
        background: var(--slate-50);
        border: 1px solid var(--slate-200);
        border-radius: 10px;
        padding: .8rem 1rem;
    }
    .info-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: .9rem 1.1rem;
        color: var(--slate-700);
        font-size: .9rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--slate-900);
        margin: 1.5rem 0 .8rem 0;
    }
    .divider {
        height: 1px;
        background: var(--slate-200);
        margin: 1.5rem 0;
        border: none;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--slate-900);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "student_name": "",
        "job_role": job_roles.JOB_ROLES[0],
        "resume_data": None,
        "github_url": "",
        "employability_answers": {},
        "match_result": None,
        "employability_result": None,
        "ml_prediction": None,
        "github_result": None,
        "skill_gap_result": None,
        "final_result": None,
        "suggestions": [],
        "evaluated": False,
        "leaderboard": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">🎓 Students Ranking System</div>', unsafe_allow_html=True)
st.sidebar.caption("AI Job Readiness Evaluation")
st.sidebar.divider()

PAGES = ["Home", "Student Details", "Resume Upload", "Employability Assessment", "Evaluation Dashboard", "Student Leaderboard", "About"]
selection = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.divider()


# ---------------------------------------------------------------------------
# Screen: Home
# ---------------------------------------------------------------------------
def render_home() -> None:
    st.markdown("## Students' Ranking System Based on Employability Estimation for Job Selection")
    st.markdown(
        "Evaluate job readiness using **resume analysis**, **employability assessment "
        "powered by machine learning**, **GitHub project credibility** and **skill gap "
        "analysis** - powered by a Random Forest model trained on real student data, "
        "running entirely on CPU with no paid APIs."
    )

    features = [
        ("📄", "Resume Analysis", "Upload a PDF resume; skills, education, projects and certifications are extracted automatically."),
        ("📊", "Employability Assessment", "Rate 8 soft-skill dimensions on a 1-5 scale. A rule-based score and ML prediction are computed."),
        ("🤖", "ML Prediction", "A Random Forest model trained on 2,900+ student records predicts Employable / LessEmployable with probabilities."),
        ("🐙", "GitHub Project Credibility", "Repository metadata is scored on activity, stars, README and language diversity."),
        ("🏆", "Student Leaderboard", "Compare readiness scores across students during the current session."),
    ]

    cols = st.columns(len(features))
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f'<div class="feature-card"><span class="icon">{icon}</span>'
                f"<h3>{title}</h3><p>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("---")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("### How it works")
        steps = [
            "Enter your name and pick a target job role.",
            "Upload your resume PDF and paste a GitHub repo URL.",
            "Rate yourself on 8 employability dimensions (1-5 scale).",
            "Click **Evaluate** to generate your Job Readiness Index.",
            "Download a PDF report and save your score to the leaderboard.",
        ]
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}.** {step}")
    with right:
        st.markdown('<div class="info-banner">', unsafe_allow_html=True)
        st.markdown("**Scoring weights**")
        st.markdown("- Employability (Rule-Based): **30%**")
        st.markdown("- ML Prediction Confidence: **20%**")
        st.markdown("- Resume Match: **25%**")
        st.markdown("- GitHub Credibility: **15%**")
        st.markdown("- Skill Completeness: **10%**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    if st.button("Start Evaluation", type="primary", use_container_width=True):
        st.info('Please continue by selecting "Student Details" from the Navigation sidebar.')


# ---------------------------------------------------------------------------
# Screen: Student Details
# ---------------------------------------------------------------------------
def render_student_details() -> None:
    st.markdown('<div class="section-title">Student Details</div>', unsafe_allow_html=True)
    st.markdown("Tell us who you are and which role you are targeting.")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Student Name", value=st.session_state.student_name, placeholder="e.g. Aarav Sharma")
    with col2:
        role = st.selectbox(
            "Target Job Role",
            job_roles.JOB_ROLES,
            index=job_roles.JOB_ROLES.index(st.session_state.job_role) if st.session_state.job_role in job_roles.JOB_ROLES else 0,
        )

    if role:
        st.markdown(f'<div class="info-banner">ℹ️ {job_roles.get_role_description(role)}</div>', unsafe_allow_html=True)

    st.session_state.student_name = name.strip()
    st.session_state.job_role = role

    st.markdown("---")
    can_continue = bool(name.strip())
    if not can_continue:
        st.warning("Please enter your name to continue.")
    if st.button("Continue to Resume Upload", type="primary", disabled=not can_continue, use_container_width=True):
        st.info("Use the sidebar to open the Resume Upload page.")


# ---------------------------------------------------------------------------
# Screen: Resume Upload
# ---------------------------------------------------------------------------
def render_resume_upload() -> None:
    st.markdown('<div class="section-title">Resume Upload</div>', unsafe_allow_html=True)
    if not st.session_state.student_name:
        st.warning("Please enter your name on the Student Details page first.")
        return

    st.markdown(f"**Student:** {st.session_state.student_name}  ·  **Target Role:** {st.session_state.job_role}")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        uploaded = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
        if uploaded is not None:
            try:
                data = resume_parser.parse_resume(uploaded.getvalue())
                st.session_state.resume_data = data
                st.success("Resume parsed successfully.")
            except ValueError as exc:
                st.session_state.resume_data = None
                st.error(str(exc))
            except Exception as exc:
                st.session_state.resume_data = None
                st.error(f"Unexpected error while parsing the PDF: {exc}")

    with col2:
        github_url = st.text_input(
            "GitHub Repository URL",
            value=st.session_state.github_url,
            placeholder="https://github.com/owner/repo",
            help="Public repository URL. Private repos cannot be analysed.",
        )
        st.session_state.github_url = github_url.strip()
        if github_url:
            parsed = github_analyzer.parse_github_url(github_url)
            if parsed is None:
                st.error("Invalid GitHub URL. Use the form https://github.com/owner/repo")
            else:
                st.caption(f"Detected: **{parsed[0]}/{parsed[1]}**")

    data = st.session_state.resume_data
    if data is not None:
        st.markdown("### Extracted Information")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Skills detected**")
            if data.skills:
                st.markdown("".join(f"`{s}` " for s in data.skills))
            else:
                st.caption("No skills detected. Consider adding a skills section to your resume.")
            st.markdown("**Education**")
            if data.education:
                st.markdown("".join(f"- {e}\n" for e in data.education))
            else:
                st.caption("No education entries detected.")
        with c2:
            st.markdown("**Certifications**")
            if data.certifications:
                st.markdown("".join(f"- {c}\n" for c in data.certifications))
            else:
                st.caption("No certifications detected.")
            st.markdown("**Projects**")
            if data.projects:
                st.markdown("".join(f"- {p}\n" for p in data.projects))
            else:
                st.caption("No projects detected.")


# ---------------------------------------------------------------------------
# Screen: Employability Assessment
# ---------------------------------------------------------------------------
def render_employability_assessment() -> None:
    st.markdown('<div class="section-title">Employability Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        "Rate yourself on each of the 8 employability dimensions below using a "
        "1-5 scale (matching the training dataset). Your ratings feed both the "
        "rule-based score and the ML model prediction."
    )

    fields = employability_form.EMPLOYABILITY_FIELDS
    answers: dict[str, int] = {}
    cols = st.columns(2)
    for i, field_def in enumerate(fields):
        with cols[i % 2]:
            current = st.session_state.employability_answers.get(field_def.key, 3)
            answers[field_def.key] = st.slider(
                field_def.label,
                min_value=employability_form.RATING_MIN,
                max_value=employability_form.RATING_MAX,
                value=current,
                help=field_def.help_text,
                key=f"emp_{field_def.key}",
            )
    st.session_state.employability_answers = {k: v for k, v in answers.items()}

    st.markdown("---")
    preview = employability_form.score_employability_rule(answers)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Preview Rule-Based Score", f"{preview.score}/100")
    with c2:
        st.progress(preview.score / 100.0)
    st.caption("This is a live preview. The rule-based score and ML prediction are finalised when you run the evaluation.")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def run_evaluation() -> bool:
    name = st.session_state.student_name.strip()
    role = st.session_state.job_role
    resume = st.session_state.resume_data
    github_url = st.session_state.github_url.strip()

    if not name:
        st.error("Student name is missing. Please fill it in on the Student Details page.")
        return False
    if resume is None:
        st.error("Resume not uploaded or parsed. Please upload a PDF on the Resume Upload page.")
        return False

    answers = st.session_state.employability_answers
    if not answers:
        st.error("Please complete the Employability Assessment form first.")
        return False

    # 1. Resume match
    match = resume_matcher.match_resume(resume.text, role)
    st.session_state.match_result = match

    # 2. Employability rule-based score
    employability = employability_form.score_employability_rule(answers)
    st.session_state.employability_result = employability

    # 3. ML prediction
    with st.spinner("Running ML prediction..."):
        ml_pred = ml_model.predict(answers)
    st.session_state.ml_prediction = ml_pred

    # 4. GitHub credibility
    if github_url and github_analyzer.parse_github_url(github_url):
        with st.spinner("Analysing GitHub repository..."):
            st.session_state.github_result = github_analyzer.analyze_github_repo(github_url)
    else:
        st.session_state.github_result = github_analyzer.GitHubCredibilityResult(
            score=0, reasons=["No valid GitHub URL provided."],
        )

    # 5. Skill gap
    gap = skill_gap.analyze_skill_gap(resume.skills, role)
    st.session_state.skill_gap_result = gap

    # 6. Final score — ml_confidence scaled to 0-100
    ml_confidence_score = int(round(ml_pred.confidence * 100))
    final = final_scoring.compute_final_score(
        employability_rule=employability.score,
        ml_confidence=ml_confidence_score,
        resume_match=match.score,
        github_credibility=st.session_state.github_result.score,
        skill_completeness=gap.completeness,
    )
    st.session_state.final_result = final

    # 7. Suggestions
    st.session_state.suggestions = resume_suggestions.generate_suggestions(
        resume, match, gap, st.session_state.github_result.has_readme,
    )

    st.session_state.evaluated = True
    return True


# ---------------------------------------------------------------------------
# Screen: Evaluation Dashboard
# ---------------------------------------------------------------------------
def render_evaluation_dashboard() -> None:
    st.markdown('<div class="section-title">Evaluation Dashboard</div>', unsafe_allow_html=True)

    col_run, col_status = st.columns([1, 2])
    with col_run:
        if st.button("Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Computing scores..."):
                ok = run_evaluation()
                if ok:
                    st.success("Evaluation complete!")
                    st.rerun()
    with col_status:
        if st.session_state.evaluated and st.session_state.final_result:
            st.markdown(
                f'<div class="info-banner">Last evaluation for '
                f'**{st.session_state.student_name}** targeting '
                f'**{st.session_state.job_role}**.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No evaluation yet. Click **Run Evaluation** after completing the inputs.")

    st.markdown("---")

    if not st.session_state.evaluated or st.session_state.final_result is None:
        st.info("Complete the previous steps and click **Run Evaluation** to see your dashboard.")
        return

    final: final_scoring.FinalScoreResult = st.session_state.final_result
    match: resume_matcher.ResumeMatchResult = st.session_state.match_result
    employability: employability_form.EmployabilityRuleResult = st.session_state.employability_result
    ml_pred: employability_form.MLPredictionResult = st.session_state.ml_prediction
    github: github_analyzer.GitHubCredibilityResult = st.session_state.github_result
    gap: skill_gap.SkillGapResult = st.session_state.skill_gap_result

    # --- Score cards row ---
    st.markdown("### Score Cards")
    cards = st.columns(4)
    score_specs = [
        ("Employability (Rule)", final.employability_rule, "#1d4ed8"),
        ("ML Prediction", ml_pred.predicted_class, "#7c3aed"),
        ("Resume Match", match.score, "#0ea5e9"),
        ("GitHub Credibility", github.score, "#16a34a"),
    ]
    for col, (label, score, color) in zip(cards, score_specs):
        with col:
            display = str(score)
            if isinstance(score, int):
                display = f"{score}"
            st.markdown(
                f'<div class="score-card"><div class="value" style="color:{color}">{display}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")

    # ML prediction details
    st.markdown("### ML Prediction Details")
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        pred_color = "#16a34a" if ml_pred.predicted_class == "Employable" else "#dc2626"
        st.markdown(
            f'<div class="info-banner" style="text-align:center;">'
            f'Predicted Class: <b style="color:{pred_color}">{ml_pred.predicted_class}</b><br>'
            f'Confidence: <b>{ml_pred.confidence*100:.1f}%</b></div>',
            unsafe_allow_html=True,
        )
    with ml_col2:
        prob_text = " · ".join(f"{cls}: **{p*100:.1f}%**" for cls, p in ml_pred.probabilities.items())
        st.markdown(
            f'<div class="info-banner">Model-estimated probabilities:<br>{prob_text}</div>',
            unsafe_allow_html=True,
        )
    st.caption("Probabilities are model-estimated, not guaranteed placement chances.")

    # Model metrics
    with st.expander("ML Model Performance (measured on test set)", expanded=False):
        try:
            metrics = ml_model.load_metrics()
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{metrics.accuracy*100:.1f}%")
            mc2.metric("Precision", f"{metrics.precision*100:.1f}%")
            mc3.metric("Recall", f"{metrics.recall*100:.1f}%")
            mc4.metric("F1 Score", f"{metrics.f1*100:.1f}%")
            st.caption(f"5-fold CV best score: {metrics.cv_best_score*100:.1f}%  ·  "
                       f"Train size: {metrics.train_size}  ·  Test size: {metrics.test_size}")
            st.text(metrics.classification_report)
        except Exception:
            st.warning("Model metrics not available.")

    level_color = {
        "Excellent": "#16a34a", "Very Good": "#0ea5e9", "Good": "#1d4ed8",
        "Average": "#f59e0b", "Needs Improvement": "#dc2626",
    }[final.level]
    st.markdown(
        f'<div class="info-banner" style="text-align:center;">'
        f'Readiness Level: <b style="color:{level_color}">{final.level}</b></div>',
        unsafe_allow_html=True,
    )

    # --- Gauge + radar ---
    st.markdown("### Readiness Index & Profile")
    gcol, rcol = st.columns([1, 1])
    with gcol:
        st.plotly_chart(charts.gauge_chart(final.score), use_container_width=True)
    with rcol:
        radar_metrics = {
            "Employability": final.employability_rule,
            "ML Confidence": final.ml_confidence,
            "Resume Match": match.score,
            "GitHub": github.score,
            "Skill Gap": gap.completeness,
        }
        st.plotly_chart(charts.radar_chart(radar_metrics), use_container_width=True)

    # --- Progress bars ---
    st.markdown("### Sub-Score Breakdown")
    pbars = st.columns(2)
    with pbars[0]:
        st.plotly_chart(charts.progress_bar_row("Employability Rule-Based (30%)", final.employability_rule), use_container_width=True)
        st.plotly_chart(charts.progress_bar_row("Resume Match (25%)", match.score), use_container_width=True)
        st.plotly_chart(charts.progress_bar_row("Skill Completeness (10%)", gap.completeness), use_container_width=True)
    with pbars[1]:
        st.plotly_chart(charts.progress_bar_row("ML Confidence (20%)", final.ml_confidence), use_container_width=True)
        st.plotly_chart(charts.progress_bar_row("GitHub Credibility (15%)", github.score), use_container_width=True)

    # --- Employability breakdown ---
    with st.expander("Employability Score Breakdown", expanded=False):
        st.plotly_chart(charts.employability_breakdown_chart(employability.breakdown), use_container_width=True)

    # --- Skill gap ---
    st.markdown("### Skill Gap Analysis")
    sg1, sg2, sg3 = st.columns(3)
    with sg1:
        st.markdown("**Existing Skills**")
        if gap.existing_skills:
            st.markdown("".join(f"`{s}` " for s in gap.existing_skills))
        else:
            st.caption("None matched yet.")
    with sg2:
        st.markdown("**Missing Skills**")
        if gap.missing_skills:
            st.markdown("".join(f"`{s}` " for s in gap.missing_skills))
        else:
            st.caption("None - you cover the required skills.")
    with sg3:
        st.markdown("**Suggested Skills to Learn**")
        if gap.suggested_skills:
            st.markdown("".join(f"`{s}` " for s in gap.suggested_skills))
        else:
            st.caption("Nothing to suggest right now.")

    # --- GitHub reasons ---
    st.markdown("### GitHub Credibility Reasons")
    if github.fetched:
        st.caption(f"Repository: **{github.repo_name}** · ⭐ {github.stars} · 🍴 {github.forks} · 🛠 {', '.join(github.languages[:4]) or 'n/a'}")
    for reason in github.reasons:
        st.markdown(f"- {reason}")

    # --- Resume suggestions ---
    st.markdown("### Resume Improvement Suggestions")
    for s in st.session_state.suggestions:
        st.markdown(f"- {s}")

    # --- Actions: PDF + leaderboard ---
    st.markdown("---")
    st.markdown("### Export & Save")
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("📄 Download PDF Report", use_container_width=True):
            try:
                pdf_bytes = pdf_report.generate_pdf_report(
                    student_name=st.session_state.student_name,
                    job_role=st.session_state.job_role,
                    resume=st.session_state.resume_data,
                    match=match,
                    employability=employability,
                    ml_prediction=ml_pred,
                    github=github,
                    skill_gap=gap,
                    final=final,
                )
                stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
                safe_name = st.session_state.student_name.replace(" ", "_") or "student"
                st.download_button(
                    label="Click to download report.pdf",
                    data=pdf_bytes,
                    file_name=f"job_readiness_{safe_name}_{stamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Could not generate the PDF: {exc}")
    with action_cols[1]:
        if st.button("🏆 Save to Leaderboard", use_container_width=True):
            entry = leaderboard.LeaderboardEntry(
                name=st.session_state.student_name,
                job_role=st.session_state.job_role,
                score=final.score,
                level=final.level,
            )
            leaderboard.add_to_leaderboard(st.session_state, entry)
            st.success(f"Saved {st.session_state.student_name}'s score ({final.score}) to the leaderboard.")
            st.rerun()


# ---------------------------------------------------------------------------
# Screen: Student Leaderboard
# ---------------------------------------------------------------------------
def render_leaderboard() -> None:
    st.markdown('<div class="section-title">Student Leaderboard</div>', unsafe_allow_html=True)
    st.markdown("Rankings for the current session. The board resets when the app reloads.")

    board = leaderboard.get_leaderboard(st.session_state)
    if not board:
        st.info("No evaluations saved yet. Run an evaluation and click **Save to Leaderboard**.")
        return

    header = st.columns([0.5, 3, 2.5, 1.5, 2])
    for col, label in zip(header, ["Rank", "Student", "Job Role", "Score", "Level"]):
        col.markdown(f"**{label}**")
    st.markdown('<hr style="margin:0.3rem 0;border:none;border-top:1px solid #e2e8f0;"/>', unsafe_allow_html=True)

    for i, entry in enumerate(board, 1):
        medal = leaderboard.medal_for_rank(i)
        row = st.columns([0.5, 3, 2.5, 1.5, 2])
        row[0].markdown(f"{medal} {i}")
        row[1].markdown(entry.name)
        row[2].markdown(entry.job_role)
        row[3].markdown(f"**{entry.score}**")
        level_color = {
            "Excellent": "#16a34a", "Very Good": "#0ea5e9", "Good": "#1d4ed8",
            "Average": "#f59e0b", "Needs Improvement": "#dc2626",
        }.get(entry.level, "#475569")
        row[4].markdown(f"<span style='color:{level_color};font-weight:600'>{entry.level}</span>", unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Clear Leaderboard"):
            leaderboard.clear_leaderboard(st.session_state)
            st.rerun()
    with c2:
        st.caption(f"{len(board)} student(s) ranked this session.")


# ---------------------------------------------------------------------------
# Screen: About
# ---------------------------------------------------------------------------
def render_about() -> None:
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)

    st.markdown("### Project Objective")
    st.markdown(
        "A modern, student-oriented web application that evaluates a student's job readiness "
        "using resume analysis, employability assessment powered by machine learning, GitHub "
        "project credibility analysis and skill gap analysis - using a Random Forest model "
        "trained on real student employability data, running on CPU without any paid LLM APIs."
    )

    st.markdown("### Workflow")
    workflow = [
        "Enter student name and select a target job role.",
        "Upload a PDF resume (parsed with PyMuPDF).",
        "Enter a public GitHub repository URL.",
        "Rate yourself on 8 employability dimensions (1-5 scale).",
        "Click Evaluate to run all analyses including ML prediction.",
        "View the dashboard, download a PDF report and save to the leaderboard.",
    ]
    for i, step in enumerate(workflow, 1):
        st.markdown(f"{i}. {step}")

    st.markdown("### ML Model")
    st.markdown(
        "The employability prediction uses a **Random Forest classifier** trained on the "
        "Student-Employability dataset (2,900+ records, 8 features) with **GridSearchCV** "
        "hyperparameter tuning and **5-fold cross-validation**. The model is trained once "
        "and saved to disk; the app loads it without retraining for each prediction."
    )

    st.markdown("### Scoring Method")
    st.markdown(
        "The **AI Job Readiness Index (0-100)** is a weighted combination of:\n\n"
        "| Component | Weight |\n|---|---|\n| Employability (Rule-Based) | 30% |\n"
        "| ML Prediction Confidence | 20% |\n| Resume Match | 25% |\n"
        "| GitHub Credibility | 15% |\n| Skill Completeness | 10% |"
    )
    st.markdown(
        "Readiness levels: **Excellent** (85+), **Very Good** (70-84), "
        "**Good** (55-69), **Average** (40-54), **Needs Improvement** (<40)."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if selection == "Home":
    render_home()
elif selection == "Student Details":
    render_student_details()
elif selection == "Resume Upload":
    render_resume_upload()
elif selection == "Employability Assessment":
    render_employability_assessment()
elif selection == "Evaluation Dashboard":
    render_evaluation_dashboard()
elif selection == "Student Leaderboard":
    render_leaderboard()
elif selection == "About":
    render_about()