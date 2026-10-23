"""Plotly chart builders for the evaluation dashboard.

Updated to reflect 8 employability dimensions instead of the old 10 daily
habits questions.  All charts use the app's blue/slate colour palette.
"""

from __future__ import annotations

import plotly.graph_objects as go


PALETTE = {
    "primary": "#1d4ed8",
    "primary_light": "#3b82f6",
    "accent": "#0ea5e9",
    "success": "#16a34a",
    "warning": "#f59e0b",
    "error": "#dc2626",
    "slate": "#475569",
    "grid": "#e2e8f0",
}


def gauge_chart(score: int, title: str = "AI Job Readiness Index") -> go.Figure:
    """Circular gauge for the final readiness index."""
    color = (
        PALETTE["success"] if score >= 70
        else PALETTE["warning"] if score >= 40
        else PALETTE["error"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 16, "color": "#1f2937"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#cbd5e1",
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 55], "color": "#fef3c7"},
                {"range": [55, 70], "color": "#dbeafe"},
                {"range": [70, 85], "color": "#bbf7d0"},
                {"range": [85, 100], "color": "#86efac"},
            ],
            "threshold": {
                "line": {"color": "#1f2937", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
    return fig


def radar_chart(metrics: dict[str, int]) -> go.Figure:
    """Radar chart across the five scoring dimensions."""
    categories = list(metrics.keys())
    values = list(metrics.values())
    values_closed = values + [values[0]] if values else []
    categories_closed = categories + [categories[0]] if categories else []

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(29, 78, 216, 0.18)",
        line={"color": PALETTE["primary"], "width": 2.5},
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], tickfont={"size": 10, "color": "#64748b"}, gridcolor=PALETTE["grid"]),
            angularaxis=dict(tickfont={"size": 12, "color": "#1f2937"}, gridcolor=PALETTE["grid"]),
            bgcolor="rgba(255,255,255,0)",
        ),
        showlegend=False,
        height=380,
        margin=dict(t=30, b=30, l=40, r=40),
    )
    return fig


def progress_bar_row(label: str, score: int, max_val: int = 100) -> go.Figure:
    """Horizontal progress bar for a single sub-score."""
    color = (
        PALETTE["success"] if score >= 70
        else PALETTE["warning"] if score >= 40
        else PALETTE["error"]
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[score],
        y=[label],
        orientation="h",
        marker=dict(color=color, line=dict(color="white", width=0)),
        text=[f"{score}/100"],
        textposition="outside",
        textfont=dict(size=13, color="#1f2937"),
        width=0.55,
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, max_val + 12], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13, color="#1f2937"), autorange="reversed"),
        height=90,
        margin=dict(t=4, b=4, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def employability_breakdown_chart(breakdown: list[dict]) -> go.Figure:
    """Horizontal bar chart showing per-field contribution to the employability score."""
    labels = [b["field"] for b in breakdown]
    values = [b["contribution"] for b in breakdown]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=PALETTE["primary_light"], line=dict(color="white", width=0)),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
        width=0.6,
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Contribution to Employability Score", range=[0, max(values) * 1.25 if values else 1], showgrid=False, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11, color="#1f2937"), autorange="reversed"),
        height=420,
        margin=dict(t=10, b=20, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig