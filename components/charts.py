"""
components/charts.py
All charts built with Plotly only, styled for the dark glass theme.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
FONT_COLOR = "#f1f5f9"
NEON_PURPLE = "#a855f7"
NEON_BLUE = "#3b82f6"
NEON_PINK = "#ec4899"
SUCCESS = "#22c55e"
DANGER = "#ef4444"


def _base_layout(fig, title=None, height=360):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        height=height,
        title=dict(text=title, font=dict(size=16, color=FONT_COLOR)) if title else None,
    )
    return fig


def gauge_chart(value: float, title: str = "Predicted Final Grade"):
    color = SUCCESS if value >= 60 else ("#f59e0b" if value >= 40 else DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number={"suffix": " / 100", "font": {"size": 34, "color": FONT_COLOR}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": FONT_COLOR},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.25)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.22)"},
                {"range": [70, 100], "color": "rgba(34,197,94,0.22)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": value},
        },
    ))
    return _base_layout(fig, title, height=320)


def feature_importance_chart(fi_df: pd.DataFrame, top_n: int = 10):
    df = fi_df.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, NEON_BLUE], [1, NEON_PINK]],
            line=dict(width=0),
        ),
    ))
    return _base_layout(fig, "Feature Importance", height=420)


def probability_chart(prob_map: dict):
    labels = list(prob_map.keys())
    values = [prob_map[k] * 100 for k in labels]
    colors = [SUCCESS if l == "Pass" else DANGER for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_yaxes(range=[0, 105], title="Probability (%)")
    return _base_layout(fig, "Prediction Probability", height=320)


def pass_fail_pie(counts: pd.Series):
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=[SUCCESS if l == "Pass" else DANGER for l in counts.index]),
        textinfo="label+percent",
    ))
    return _base_layout(fig, "Pass vs Fail Distribution", height=360)


def actual_vs_predicted_chart(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=NEON_PURPLE, size=6, opacity=0.55),
        name="Students",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines",
        line=dict(color=NEON_PINK, dash="dash", width=2),
        name="Perfect Prediction",
    ))
    fig.update_xaxes(title="Actual Final Grade", range=[0, 100])
    fig.update_yaxes(title="Predicted Final Grade", range=[0, 100])
    return _base_layout(fig, "Actual vs Predicted Grade", height=420)


def confusion_matrix_chart(y_true, y_pred, labels=("Pass", "Fail")):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Predicted {l}" for l in labels],
        y=[f"Actual {l}" for l in labels],
        colorscale=[[0, "rgba(59,130,246,0.1)"], [1, NEON_PURPLE]],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "white"},
        showscale=False,
    ))
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, "Confusion Matrix", height=380)
