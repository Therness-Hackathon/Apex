"""
Apex Weld Quality – Interactive Dashboard
Run:  python dashboard_app.py
Then open  http://127.0.0.1:8050  in your browser.
"""

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DASH_IMG = OUTPUTS / "dashboard"

# ─── Load Data ────────────────────────────────────────────────────────────────
manifest   = pd.read_csv(OUTPUTS / "manifest.csv")
training   = pd.read_csv(OUTPUTS / "training_history.csv")
preds_test = pd.read_csv(OUTPUTS / "predictions_binary.csv")
preds_val  = pd.read_csv(OUTPUTS / "predictions_binary_val.csv")

# ─── Computed Stats ───────────────────────────────────────────────────────────
THRESHOLD = 0.34
ROC_AUC   = 0.8827
AVG_PREC  = 0.8283

total_runs    = len(manifest)
good_count    = int((manifest["label"] == 0).sum())
defect_count  = int((manifest["label"] == 1).sum())
defect_types  = (
    manifest[manifest["label"] == 1]["defect_type"]
    .value_counts()
    .sort_values()
)

splits = {"train": 1047, "val": 240, "test": 264}

# Live-compute test metrics from the binary prediction file
t = preds_test.copy()
t["pred"] = (t["p_defect"] >= THRESHOLD).astype(int)
TP  = int(((t["pred"] == 1) & (t["label"] == 1)).sum())
FP  = int(((t["pred"] == 1) & (t["label"] == 0)).sum())
TN  = int(((t["pred"] == 0) & (t["label"] == 0)).sum())
FN  = int(((t["pred"] == 0) & (t["label"] == 1)).sum())
acc  = (TP + TN) / len(t)
prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

# ─── Design tokens ────────────────────────────────────────────────────────────
BG        = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
GREEN     = "#3fb950"
RED       = "#f85149"
BLUE      = "#58a6ff"
ORANGE    = "#ffa657"
PURPLE    = "#d2a8ff"
YELLOW    = "#e3b341"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
TEMPLATE  = "plotly_dark"

PALETTE = [GREEN, ORANGE, BLUE, YELLOW, PURPLE, RED, "#79c0ff"]

# ─── Reusable components ──────────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str = "", color: str = GREEN):
    return dbc.Card(
        dbc.CardBody([
            html.P(label, style={
                "fontSize": "0.72rem", "textTransform": "uppercase",
                "letterSpacing": "1.5px", "color": MUTED, "marginBottom": "4px"
            }),
            html.H3(value, style={
                "color": color, "fontWeight": "700",
                "margin": "0 0 2px 0", "lineHeight": "1"
            }),
            html.P(sub, style={"fontSize": "0.73rem", "color": MUTED, "margin": 0}),
        ]),
        style={
            "background": CARD_BG, "border": f"1px solid {color}44",
            "borderRadius": "10px", "height": "100%"
        },
        className="shadow-sm",
    )


def section(title: str):
    return html.P(title, style={
        "fontSize": "0.68rem", "textTransform": "uppercase",
        "letterSpacing": "2px", "color": MUTED,
        "borderBottom": f"1px solid {BORDER}", "paddingBottom": "6px",
        "marginBottom": "12px"
    })


def panel(*children, height: str | None = None):
    style = {"background": CARD_BG, "border": f"1px solid {BORDER}",
             "borderRadius": "10px", "height": "100%"}
    if height:
        style["minHeight"] = height
    return dbc.Card(dbc.CardBody(list(children)), style=style, className="shadow-sm")


GRAPH_CFG: Any = {"displayModeBar": False}

# ─── Figures ──────────────────────────────────────────────────────────────────

def _base(fig, h: int = 280, mt: int = 10, ml: int = 50, mr: int = 20, mb: int = 40):
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=TEXT, size=11)),
        margin=dict(t=mt, b=mb, l=ml, r=mr),
        height=h,
        font=dict(color=TEXT),
    )
    return fig


def fig_dataset_pie():
    fig = go.Figure(go.Pie(
        labels=["Good", "Defect"],
        values=[good_count, defect_count],
        hole=0.58,
        marker=dict(colors=[GREEN, RED]),
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        pull=[0.04, 0],
    ))
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=TEXT, size=11)),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        annotations=[dict(
            text=f"<b>{total_runs}</b><br><span style='font-size:10px'>Runs</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=TEXT),
        )],
    )
    return fig


def fig_split_bar():
    fig = go.Figure(go.Bar(
        x=list(splits.keys()),
        y=list(splits.values()),
        marker_color=[BLUE, YELLOW, ORANGE],
        text=list(splits.values()),
        textposition="outside",
        textfont=dict(color=TEXT),
        hovertemplate="%{x}: %{y} samples<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=30, l=40, r=20),
        height=200,
        xaxis=dict(tickfont=dict(size=12, color=TEXT)),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
        font=dict(color=TEXT),
    )
    return fig


def fig_defect_bar():
    labels = [l.replace("_", " ").title() for l in defect_types.index]
    vals   = defect_types.values.tolist()
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker=dict(color=PALETTE[:len(labels)], opacity=0.88),
        text=vals, textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    _base(fig, h=300, mt=10, ml=10, mr=50, mb=20)
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=10.5)),
    )
    return fig


def fig_training_loss():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["train_loss"],
        mode="lines+markers", name="Train Loss",
        line=dict(color=BLUE, width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["val_loss"],
        mode="lines+markers", name="Val Loss",
        line=dict(color=RED, width=2, dash="dot"), marker=dict(size=5),
    ))
    _base(fig, h=260)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    return fig


def fig_val_scores():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["val_acc"],
        mode="lines+markers", name="Accuracy",
        line=dict(color=GREEN, width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["val_f1"],
        mode="lines+markers", name="F1 Score",
        line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=5),
    ))
    _base(fig, h=260)
    fig.update_layout(
        xaxis_title="Epoch", yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def fig_confusion_matrix():
    z    = [[TN, FP], [FN, TP]]
    text = [[f"TN<br>{TN}", f"FP<br>{FP}"], [f"FN<br>{FN}", f"TP<br>{TP}"]]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: Good", "Pred: Defect"],
        y=["True: Good", "True: Defect"],
        colorscale=[[0, "#1f2937"], [1, GREEN]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=15, color="white"),
        showscale=False,
        hovertemplate="%{x} / %{y}: %{z}<extra></extra>",
    ))
    _base(fig, h=280, ml=100, mb=50)
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    return fig


def fig_pred_dist():
    good_p   = preds_test[preds_test["label"] == 0]["p_defect"]
    defect_p = preds_test[preds_test["label"] == 1]["p_defect"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=good_p, name="True Good", nbinsx=30,
        marker_color=GREEN, opacity=0.70,
    ))
    fig.add_trace(go.Histogram(
        x=defect_p, name="True Defect", nbinsx=30,
        marker_color=RED, opacity=0.70,
    ))
    fig.add_vline(
        x=THRESHOLD, line_dash="dash", line_color=YELLOW, line_width=2,
        annotation_text=f"  Threshold={THRESHOLD}",
        annotation_font_color=YELLOW, annotation_font_size=11,
    )
    _base(fig, h=280)
    fig.update_layout(
        barmode="overlay",
        xaxis_title="P(Defect)",
        yaxis_title="Count",
    )
    return fig


def fig_duration_hist():
    good_d   = manifest[manifest["label"] == 0]["duration_s"]
    defect_d = manifest[manifest["label"] == 1]["duration_s"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=good_d, name="Good", nbinsx=40,
        marker_color=GREEN, opacity=0.70,
    ))
    fig.add_trace(go.Histogram(
        x=defect_d, name="Defect", nbinsx=40,
        marker_color=RED, opacity=0.70,
    ))
    _base(fig, h=260)
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Duration (s)", yaxis_title="Count",
    )
    return fig


def fig_confidence_box():
    """Box plot of confidence scores split by correct / incorrect predictions."""
    t2 = preds_test.copy()
    t2["pred"]    = (t2["p_defect"] >= THRESHOLD).astype(int)
    t2["correct"] = t2["pred"] == t2["label"]
    correct   = t2[t2["correct"]]["confidence"]
    incorrect = t2[~t2["correct"]]["confidence"]
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=correct, name="Correct", marker_color=GREEN,
        boxmean="sd", line_width=2,
        hovertemplate="Correct<br>%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Box(
        y=incorrect, name="Incorrect", marker_color=RED,
        boxmean="sd", line_width=2,
        hovertemplate="Incorrect<br>%{y:.3f}<extra></extra>",
    ))
    _base(fig, h=260, ml=40, mr=20)
    fig.update_layout(yaxis_title="Confidence Score")
    return fig


# ─── Metrics table rows ────────────────────────────────────────────────────────
_ROW_STYLE = {"fontSize": "0.82rem"}


def _metric_row(label, val_v, test_v, highlight=False):
    style = {"color": GREEN} if highlight else {}
    return html.Tr([
        html.Td(label, style={"color": MUTED, "fontSize": "0.8rem"}),
        html.Td(val_v),
        html.Td(test_v, style=style),
    ])


# ─── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP],
    title="Apex Weld Quality Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        # ── Header ────────────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        html.Div([
                            html.Span("⚡", style={"fontSize": "1.8rem", "marginRight": "10px"}),
                            html.H1(
                                "Apex Weld Quality",
                                style={"color": GREEN, "fontWeight": "700",
                                       "margin": 0, "display": "inline", "fontSize": "1.8rem"},
                            ),
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.P(
                            "AI-Powered Weld Classification  ·  Phase 2  ·  Binary Defect Detection",
                            style={"color": MUTED, "margin": "2px 0 0 38px", "fontSize": "0.85rem"},
                        ),
                    ]),
                    width=9,
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Badge("PHASE 2", color="success", pill=True, className="me-2"),
                            html.Span(
                                f"Threshold: {THRESHOLD}  |  Temp: 1.194",
                                style={"color": MUTED, "fontSize": "0.78rem"},
                            ),
                        ],
                        className="d-flex align-items-center justify-content-end h-100",
                    ),
                    width=3,
                ),
            ],
            className="py-3 mb-3",
            style={"borderBottom": f"1px solid {BORDER}"},
        ),

        # ── KPI Row ───────────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(kpi_card("Total Runs",    f"{total_runs:,}",   "dataset samples",     TEXT),   width=2),
                dbc.Col(kpi_card("Good Welds",    f"{good_count:,}",   f"{good_count/total_runs:.0%}",  GREEN),  width=2),
                dbc.Col(kpi_card("Defects",       f"{defect_count:,}", f"{defect_count/total_runs:.0%}", RED),   width=2),
                dbc.Col(kpi_card("Test Accuracy", f"{acc:.1%}",        f"@ threshold {THRESHOLD}",     BLUE),   width=2),
                dbc.Col(kpi_card("F1 Score",      f"{f1:.4f}",         "Test set",                     ORANGE), width=2),
                dbc.Col(kpi_card("ROC AUC",       f"{ROC_AUC:.4f}",    "Test set",                     PURPLE), width=2),
            ],
            className="mb-4 g-3",
        ),

        # ── Section: Dataset ──────────────────────────────────────────────────
        html.P("Dataset Overview",
               style={"color": MUTED, "fontSize": "0.68rem", "textTransform": "uppercase",
                      "letterSpacing": "2.5px", "marginBottom": "10px"}),
        dbc.Row(
            [
                dbc.Col(
                    panel(
                        section("Good vs Defect"),
                        dcc.Graph(figure=fig_dataset_pie(), config=GRAPH_CFG),
                        html.Div([
                            dbc.Badge(f"Good: {good_count}", color="success", pill=True, className="me-2"),
                            dbc.Badge(f"Defect: {defect_count}", color="danger", pill=True),
                        ], className="text-center mt-1"),
                    ),
                    width=3,
                ),
                dbc.Col(
                    panel(
                        section("Defect Type Breakdown"),
                        dcc.Graph(figure=fig_defect_bar(), config=GRAPH_CFG),
                    ),
                    width=5,
                ),
                dbc.Col(
                    panel(
                        section("Train / Val / Test Split"),
                        dcc.Graph(figure=fig_split_bar(), config=GRAPH_CFG),
                        dbc.Row([
                            dbc.Col(html.Div([
                                html.P("Train", style={"color": BLUE, "fontWeight": "600", "margin": 0, "fontSize": "0.8rem"}),
                                html.P("1 047 runs", style={"color": MUTED, "margin": 0, "fontSize": "0.75rem"}),
                            ]), width=4, className="text-center"),
                            dbc.Col(html.Div([
                                html.P("Val", style={"color": YELLOW, "fontWeight": "600", "margin": 0, "fontSize": "0.8rem"}),
                                html.P("240 runs", style={"color": MUTED, "margin": 0, "fontSize": "0.75rem"}),
                            ]), width=4, className="text-center"),
                            dbc.Col(html.Div([
                                html.P("Test", style={"color": ORANGE, "fontWeight": "600", "margin": 0, "fontSize": "0.8rem"}),
                                html.P("264 runs", style={"color": MUTED, "margin": 0, "fontSize": "0.75rem"}),
                            ]), width=4, className="text-center"),
                        ], className="mt-2"),
                    ),
                    width=4,
                ),
            ],
            className="mb-4 g-3",
        ),

        # ── Section: Training ─────────────────────────────────────────────────
        html.P("Training History",
               style={"color": MUTED, "fontSize": "0.68rem", "textTransform": "uppercase",
                      "letterSpacing": "2.5px", "marginBottom": "10px"}),
        dbc.Row(
            [
                dbc.Col(
                    panel(
                        section("Loss Curves"),
                        dcc.Graph(figure=fig_training_loss(), config=GRAPH_CFG),
                    ),
                    width=6,
                ),
                dbc.Col(
                    panel(
                        section("Validation Accuracy & F1"),
                        dcc.Graph(figure=fig_val_scores(), config=GRAPH_CFG),
                    ),
                    width=6,
                ),
            ],
            className="mb-4 g-3",
        ),

        # ── Section: Model Performance ────────────────────────────────────────
        html.P("Model Performance",
               style={"color": MUTED, "fontSize": "0.68rem", "textTransform": "uppercase",
                      "letterSpacing": "2.5px", "marginBottom": "10px"}),
        dbc.Row(
            [
                # Confusion Matrix
                dbc.Col(
                    panel(
                        section("Confusion Matrix (Test)"),
                        dcc.Graph(figure=fig_confusion_matrix(), config=GRAPH_CFG),
                        dbc.Row(
                            [
                                dbc.Col(html.Div([
                                    html.Span("Precision", style={"color": MUTED, "fontSize": "0.72rem"}),
                                    html.H5(f"{prec:.3f}", style={"color": GREEN, "margin": 0}),
                                ], className="text-center"), width=3),
                                dbc.Col(html.Div([
                                    html.Span("Recall", style={"color": MUTED, "fontSize": "0.72rem"}),
                                    html.H5(f"{rec:.3f}", style={"color": ORANGE, "margin": 0}),
                                ], className="text-center"), width=3),
                                dbc.Col(html.Div([
                                    html.Span("Specificity", style={"color": MUTED, "fontSize": "0.72rem"}),
                                    html.H5(f"{spec:.3f}", style={"color": BLUE, "margin": 0}),
                                ], className="text-center"), width=3),
                                dbc.Col(html.Div([
                                    html.Span("F1", style={"color": MUTED, "fontSize": "0.72rem"}),
                                    html.H5(f"{f1:.3f}", style={"color": PURPLE, "margin": 0}),
                                ], className="text-center"), width=3),
                            ],
                            className="mt-2",
                        ),
                    ),
                    width=4,
                ),
                # Prediction Distribution
                dbc.Col(
                    panel(
                        section("Prediction Confidence Distribution (Test)"),
                        dcc.Graph(figure=fig_pred_dist(), config=GRAPH_CFG),
                    ),
                    width=8,
                ),
            ],
            className="mb-4 g-3",
        ),

        # ── Section: Deep Dive ────────────────────────────────────────────────
        html.P("Deep Dive",
               style={"color": MUTED, "fontSize": "0.68rem", "textTransform": "uppercase",
                      "letterSpacing": "2.5px", "marginBottom": "10px"}),
        dbc.Row(
            [
                dbc.Col(
                    panel(
                        section("Weld Duration Distribution"),
                        dcc.Graph(figure=fig_duration_hist(), config=GRAPH_CFG),
                    ),
                    width=4,
                ),
                dbc.Col(
                    panel(
                        section("Confidence: Correct vs Incorrect"),
                        dcc.Graph(figure=fig_confidence_box(), config=GRAPH_CFG),
                    ),
                    width=4,
                ),
                # Full metrics table
                dbc.Col(
                    panel(
                        section("Val vs Test Metrics"),
                        dbc.Table(
                            [
                                html.Thead(html.Tr([
                                    html.Th("Metric",         style={"color": MUTED, "fontSize": "0.75rem", "fontWeight": "400"}),
                                    html.Th("Validation",     style={"color": MUTED, "fontSize": "0.75rem", "fontWeight": "400"}),
                                    html.Th("Test ✓",         style={"color": GREEN, "fontSize": "0.75rem", "fontWeight": "600"}),
                                ])),
                                html.Tbody([
                                    _metric_row("Accuracy",       "81.25 %", "87.50 %", True),
                                    _metric_row("Precision",      "74.56 %", "81.33 %", True),
                                    _metric_row("Recall",         "98.44 %", "98.54 %"),
                                    _metric_row("Specificity",    "61.61 %", "75.59 %", True),
                                    _metric_row("F1 Score",       "0.8485",  "0.8911",  True),
                                    _metric_row("ROC AUC",        "0.8073",  "0.8827",  True),
                                    _metric_row("Avg Precision",  "0.7482",  "0.8283",  True),
                                    _metric_row("Log Loss",       "0.5144",  "0.4444",  True),
                                    _metric_row("Brier Score",    "0.1647",  "0.1192",  True),
                                    _metric_row("TP / FN",        "126 / 2", f"{TP} / {FN}"),
                                    _metric_row("TN / FP",        "69 / 43", f"{TN} / {FP}"),
                                ]),
                            ],
                            bordered=False, hover=True, responsive=True, size="sm",
                            style={"color": TEXT, "fontSize": "0.82rem"},
                        ),
                        html.P(
                            f"Threshold: {THRESHOLD}  ·  Temperature scaling: 1.1940",
                            style={"color": MUTED, "fontSize": "0.7rem", "marginTop": "8px", "marginBottom": 0},
                        ),
                    ),
                    width=4,
                ),
            ],
            className="mb-4 g-3",
        ),

        # ── Footer ────────────────────────────────────────────────────────────
        html.Hr(style={"borderColor": BORDER}),
        html.P(
            "Apex Weld Quality  ·  AI Classification System  ·  Phase 2  ·  Binary Defect Detection",
            className="text-center mb-3",
            style={"color": MUTED, "fontSize": "0.73rem"},
        ),
    ],
    fluid=True,
    style={"background": BG, "minHeight": "100vh", "padding": "0 28px"},
)

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Apex Weld Quality Dashboard")
    print("  Open  http://127.0.0.1:8050  in your browser\n")
    app.run(debug=False, port=8050)
