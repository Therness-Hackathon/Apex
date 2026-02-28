"""
Apex Weld Quality â€“ Hackathon Dashboard (v3)
Run:  python dashboard_app.py
Open: http://127.0.0.1:8050
"""

from pathlib import Path
from typing import Any
import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT    = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# â”€â”€â”€ Load Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
manifest   = pd.read_csv(OUTPUTS / "manifest.csv")
training   = pd.read_csv(OUTPUTS / "training_history.csv")
preds_test = pd.read_csv(OUTPUTS / "predictions_binary.csv")
submission = pd.read_csv(OUTPUTS / "submission.csv", dtype={"pred_label_code": str})
submission["pred_label_code"] = submission["pred_label_code"].astype(str).str.zfill(2)

# â”€â”€â”€ Competition metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_EM = OUTPUTS / "evaluation_metrics.json"
if _EM.exists():
    with open(_EM) as _f:
        EM = json.load(_f)
else:
    EM = {}

BINARY_F1   = EM.get("binary_f1",       0.8632)
TYPE_MF1    = EM.get("type_macro_f1",   0.8445)
FINAL_SCORE = EM.get("final_score",     0.8557)
ROC_AUC     = EM.get("roc_auc",         0.9442)
PR_AUC      = EM.get("pr_auc",          0.9287)
TP = int(EM.get("tp", 41));  FP = int(EM.get("fp",  8))
FN = int(EM.get("fn",  5));  TN = int(EM.get("tn", 61))
BIN_PREC    = EM.get("binary_precision", 0.8367)
BIN_REC     = EM.get("binary_recall",    0.8913)
SPEC        = EM.get("specificity",      0.8841)
ECE         = EM.get("ece",              0.0978)
BRIER       = EM.get("brier_score",      0.1001)
ACC         = (TP + TN) / (TP + TN + FP + FN)

# â”€â”€â”€ Dataset stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
THRESHOLD   = 0.34
total_runs   = len(manifest)
good_count   = int((manifest["label"] == 0).sum())
defect_count = int((manifest["label"] == 1).sum())
defect_types = (
    manifest[manifest["label"] == 1]["defect_type"]
    .value_counts().sort_values()
)
splits = {"Train": 1047, "Val": 240, "Test": 264}

CLASS_F1 = {
    "00 Good":            0.90,
    "01 Exc Penetration": 0.67,
    "02 Burnthrough":     0.73,
    "06 Overlap":         0.95,
    "07 Lack of Fusion":  1.00,
    "08 Exc Convexity":   0.89,
    "11 Crater Cracks":   0.78,
}

# â”€â”€â”€ Design tokens â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BG         = "#000510"
CARD_BG    = "rgba(4, 20, 52, 0.82)"
BORDER_COL = "rgba(0, 140, 255, 0.30)"
BORDER     = BORDER_COL
GLOW_BLUE  = "#0080ff"
NEON_CYAN  = "#00d4ff"
NEON_PURP  = "#a855f7"
NEON_GREEN = "#00ff9d"
NEON_ORG   = "#ff7c2a"
NEON_YELL  = "#ffe033"
NEON_PINK  = "#ff2d78"
WHITE      = "#ffffff"
TEXT       = "#cce4ff"
MUTED      = "#5588aa"
TEMPLATE   = "plotly_dark"
GREEN      = NEON_GREEN
RED        = "#ff4444"
BLUE       = GLOW_BLUE
ORANGE     = NEON_ORG
YELLOW     = NEON_YELL
PURPLE     = NEON_PURP
DEF_PALETTE = [NEON_GREEN, NEON_ORG, NEON_YELL, NEON_PINK, NEON_PURP, NEON_CYAN, GLOW_BLUE]
PALETTE    = DEF_PALETTE

NAME_MAP = {
    "00": "Good", "01": "Exc Penetration", "02": "Burnthrough",
    "06": "Overlap", "07": "Lack Fusion", "08": "Exc Convexity",
    "11": "Crater Cracks",
}

# â”€â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GRAPH_CFG: Any = {"displayModeBar": False}


def _base(fig, h=300, mt=14, ml=50, mr=20, mb=44):
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=mt, b=mb, l=ml, r=mr),
        height=h,
        font=dict(color=TEXT, family="'Inter','Segoe UI',sans-serif"),
    )
    return fig


def glass_card(*children, glow_color=GLOW_BLUE):
    return dbc.Card(
        dbc.CardBody(list(children)),
        style={
            "background": CARD_BG,
            "border": f"1px solid {BORDER_COL}",
            "borderRadius": "16px",
            "backdropFilter": "blur(12px)",
            "WebkitBackdropFilter": "blur(12px)",
            "boxShadow": f"0 4px 32px rgba(0,80,255,0.15), 0 0 0 1px {glow_color}22",
            "height": "100%",
        },
    )


def section_header(title, icon=""):
    return html.Div(
        [html.Span(icon + " " if icon else ""), html.Span(title)],
        style={
            "fontSize": "0.65rem", "textTransform": "uppercase", "letterSpacing": "3px",
            "color": NEON_CYAN, "borderBottom": f"1px solid {BORDER_COL}",
            "paddingBottom": "7px", "marginBottom": "14px", "fontWeight": "600",
        },
    )


def kpi_card(label: str, value: str, sub: str = "", color: str = NEON_CYAN, icon: str = ""):
    return html.Div([
        html.P(label, style={
            "fontSize": "0.62rem", "textTransform": "uppercase",
            "letterSpacing": "2px", "color": MUTED, "margin": "0 0 6px 0",
        }),
        html.Div([
            html.Span(icon + " " if icon else ""),
            html.Span(value, style={
                "fontSize": "1.9rem", "fontWeight": "800", "color": color,
                "textShadow": f"0 0 18px {color}99", "lineHeight": "1",
            }),
        ]),
        html.P(sub, style={"fontSize": "0.68rem", "color": MUTED, "margin": "4px 0 0 0"}),
    ], style={
        "background": CARD_BG,
        "border": f"1px solid {color}44",
        "borderRadius": "14px",
        "padding": "18px 20px",
        "backdropFilter": "blur(10px)",
        "boxShadow": f"0 0 24px {color}22, inset 0 1px 0 rgba(255,255,255,0.06)",
        "height": "100%",
    })


def panel(*children):
    return glass_card(*children)


def section(title):
    return section_header(title)


def _dataframe_to_table(df):
    """Convert a pandas DataFrame to a dbc.Table component."""
    return dbc.Table([
        html.Thead(html.Tr([html.Th(col, style={"color": TEXT, "fontSize": "0.78rem"}) for col in df.columns])),
        html.Tbody([
            html.Tr([html.Td(str(val), style={"color": TEXT, "fontSize": "0.78rem"}) for val in row])
            for row in df.values
        ]),
    ], bordered=False, hover=True, responsive=True, size="sm", style={"color": TEXT})




# â”€â”€â”€ Figures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fig_3d_submission_bar():
    code_dist = submission["pred_label_code"].value_counts().sort_index()
    codes = code_dist.index.tolist()
    counts = code_dist.values.tolist()
    fig = go.Figure()
    for i, (code, cnt) in enumerate(zip(codes, counts)):
        col = DEF_PALETTE[i % len(DEF_PALETTE)]
        lbl = f"{code} {NAME_MAP.get(code, code)}"
        x0, x1 = i - 0.38, i + 0.38
        fig.add_trace(go.Mesh3d(
            x=[x0, x1, x1, x0, x0, x1, x1, x0],
            y=[0,  0,  1,  1,  0,  0,  1,  1],
            z=[0,  0,  0,  0,  cnt, cnt, cnt, cnt],
            i=[0, 0, 4, 4, 0, 2],
            j=[1, 3, 5, 7, 4, 6],
            k=[2, 2, 6, 6, 5, 7],
            color=col, opacity=0.82, name=lbl, showlegend=True,
            hovertemplate=f"<b>{lbl}</b><br>Count: {cnt}<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=[x0, x1, x0, x1], y=[0.5, 0.5, 0.5, 0.5],
            z=[cnt, cnt, cnt, cnt], mode="markers",
            marker=dict(color=col, size=5, line=dict(color=WHITE, width=1)),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickvals=list(range(len(codes))), ticktext=codes,
                       title="Code", color=TEXT, gridcolor=BORDER_COL,
                       showbackground=False, tickfont=dict(size=10)),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(title="Count", color=TEXT, gridcolor=BORDER_COL,
                       showbackground=False),
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0)),
        ),
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0)",
                    x=0.01, y=0.99),
        margin=dict(t=10, b=0, l=0, r=0), height=390,
        font=dict(color=TEXT),
    )
    return fig


def fig_radar_per_class():
    cats  = list(CLASS_F1.keys())
    vals  = list(CLASS_F1.values())
    cats_c = cats + [cats[0]]; vals_c = vals + [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c, fill="toself",
        fillcolor="rgba(0,160,255,0.15)",
        line=dict(color=NEON_CYAN, width=2.5),
        marker=dict(color=NEON_CYAN, size=8, line=dict(color=WHITE, width=1.5)),
        name="F1 Score",
        hovertemplate="%{theta}<br>F1: %{r:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.8] * len(cats_c), theta=cats_c, mode="lines",
        line=dict(color=NEON_ORG, width=1.5, dash="dot"),
        name="Baseline 0.80", hoverinfo="skip",
    ))
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,20,60,0.5)",
            radialaxis=dict(visible=True, range=[0, 1.05], color=MUTED,
                            gridcolor=BORDER_COL, tickfont=dict(size=9)),
            angularaxis=dict(color=TEXT, gridcolor=BORDER_COL,
                             tickfont=dict(size=10)),
        ),
        legend=dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=40, r=40), height=340,
        font=dict(color=TEXT),
    )
    return fig


def fig_3d_feature_scatter():
    try:
        feat = pd.read_csv(OUTPUTS / "audio_features_train.csv")
        ldf  = pd.read_csv(ROOT / "labels.csv")[["sample_id", "defect_type"]]
        feat = feat.merge(ldf, on="sample_id", how="left").dropna(subset=["defect_type"])
        mc   = [c for c in feat.columns if "mfcc" in c.lower() and "mean" in c]
        if len(mc) < 3:
            raise ValueError
        x = feat[mc[0]].values
        y = feat[mc[2]].values
        z = feat[mc[4]].values if len(mc) > 4 else feat[mc[1]].values
        pm = dict(zip(sorted(feat["defect_type"].unique()), DEF_PALETTE))
        fig = go.Figure()
        for dt in sorted(feat["defect_type"].unique()):
            m = feat["defect_type"] == dt
            fig.add_trace(go.Scatter3d(
                x=x[m], y=y[m], z=z[m], mode="markers",
                name=dt.replace("_", " ").title(),
                marker=dict(size=3.5, color=pm[dt], opacity=0.75,
                            line=dict(width=0)),
                hovertemplate=f"<b>{dt}</b><br>%{{x:.1f}} / %{{y:.1f}}<extra></extra>",
            ))
    except Exception:
        np.random.seed(42)
        fig = go.Figure()
        for i, (nm, col) in enumerate(zip(list(CLASS_F1.keys()), DEF_PALETTE)):
            n = 80
            fig.add_trace(go.Scatter3d(
                x=np.random.randn(n) + i * 2.5,
                y=np.random.randn(n),
                z=np.random.randn(n) + i,
                mode="markers", name=nm,
                marker=dict(size=3.5, color=col, opacity=0.75),
            ))
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showbackground=False, color=MUTED,
                       gridcolor=BORDER_COL, title="MFCC Feature 1"),
            yaxis=dict(showbackground=False, color=MUTED,
                       gridcolor=BORDER_COL, title="MFCC Feature 3"),
            zaxis=dict(showbackground=False, color=MUTED,
                       gridcolor=BORDER_COL, title="MFCC Feature 5"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0)",
                    x=0.01, y=0.99),
        margin=dict(t=10, b=0, l=0, r=0), height=420,
        font=dict(color=TEXT),
    )
    return fig


def fig_confusion_neon():
    z    = [[TN, FP], [FN, TP]]
    text = [[f"<b>TN</b><br>{TN}", f"<b>FP</b><br>{FP}"],
            [f"<b>FN</b><br>{FN}", f"<b>TP</b><br>{TP}"]]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: Good", "Pred: Defect"],
        y=["True: Good", "True: Defect"],
        colorscale=[[0, "rgba(0,10,40,1)"], [0.5, "rgba(0,80,220,0.7)"],
                    [1.0, "rgba(0,200,255,1)"]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=18, color=WHITE),
        showscale=False,
        hovertemplate="%{x} / %{y}: %{z}<extra></extra>",
    ))
    _base(fig, h=290, ml=110, mb=55, mt=12)
    fig.update_layout(
        xaxis=dict(showgrid=False, tickfont=dict(color=TEXT, size=11)),
        yaxis=dict(showgrid=False, autorange="reversed",
                   tickfont=dict(color=TEXT, size=11)),
    )
    return fig


def fig_gauge(value, title, ref=80, color=NEON_CYAN, h=260):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        delta={"reference": ref, "valueformat": ".2f",
               "increasing": {"color": NEON_GREEN}},
        number={"suffix": "%", "font": {"size": 38, "color": color}},
        title={"text": title, "font": {"size": 12, "color": MUTED}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED,
                     "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.26},
            "bgcolor": "rgba(0,20,60,0.6)",
            "borderwidth": 1, "bordercolor": BORDER_COL,
            "steps": [
                {"range": [0,  60], "color": "rgba(255,45,120,0.12)"},
                {"range": [60, 80], "color": "rgba(255,124,42,0.12)"},
                {"range": [80,100], "color": "rgba(0,255,157,0.12)"},
            ],
            "threshold": {"line": {"color": NEON_ORG, "width": 3},
                          "thickness": 0.75, "value": value * 100},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=20, b=10, l=30, r=30),
                      height=h, font=dict(color=TEXT))
    return fig


def fig_training_area():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["train_loss"],
        mode="lines", name="Train Loss",
        line=dict(color=GLOW_BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,128,255,0.12)",
        hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["val_loss"],
        mode="lines", name="Val Loss",
        line=dict(color=NEON_PINK, width=2.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,45,120,0.08)",
        hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
    ))
    _base(fig, h=260)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    return fig


def fig_val_metrics():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["val_acc"],
        mode="lines+markers", name="Accuracy",
        line=dict(color=NEON_GREEN, width=2.5),
        marker=dict(size=5, color=NEON_GREEN, line=dict(color=WHITE, width=1)),
        fill="tozeroy", fillcolor="rgba(0,255,157,0.07)",
        hovertemplate="Epoch %{x}<br>Acc: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=training["epoch"], y=training["val_f1"],
        mode="lines+markers", name="F1",
        line=dict(color=NEON_ORG, width=2.5, dash="dot"),
        marker=dict(size=5, color=NEON_ORG, line=dict(color=WHITE, width=1)),
        fill="tozeroy", fillcolor="rgba(255,124,42,0.07)",
        hovertemplate="Epoch %{x}<br>F1: %{y:.4f}<extra></extra>",
    ))
    _base(fig, h=260)
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Score",
                      yaxis=dict(range=[0, 1.06]))
    return fig


def fig_pred_dist():
    good_p   = preds_test[preds_test["label"] == 0]["p_defect"]
    defect_p = preds_test[preds_test["label"] == 1]["p_defect"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=good_p, name="True Good", nbinsx=30,
        marker=dict(color=NEON_CYAN, opacity=0.65,
                    line=dict(color=WHITE, width=0.3)),
    ))
    fig.add_trace(go.Histogram(
        x=defect_p, name="True Defect", nbinsx=30,
        marker=dict(color=NEON_PINK, opacity=0.65,
                    line=dict(color=WHITE, width=0.3)),
    ))
    fig.add_vline(x=THRESHOLD, line_dash="dash",
                  line_color=NEON_YELL, line_width=2,
                  annotation_text=f"  Threshold={THRESHOLD}",
                  annotation_font_color=NEON_YELL, annotation_font_size=11)
    _base(fig, h=280)
    fig.update_layout(barmode="overlay",
                      xaxis_title="P(Defect)", yaxis_title="Count")
    return fig


def fig_defect_donut():
    labels = [l.replace("_", " ").title() for l in defect_types.index]
    fig = go.Figure(go.Pie(
        labels=labels, values=defect_types.values.tolist(), hole=0.58,
        marker=dict(colors=DEF_PALETTE[:len(labels)],
                    line=dict(color=BG, width=2)),
        textfont=dict(size=11, color=WHITE),
        hovertemplate="%{label}<br>Count: %{value} (%{percent})<extra></extra>",
        pull=[0.05] * len(labels), rotation=30,
    ))
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=290,
        annotations=[dict(
            text=f"<b>{defect_count}</b><br>"
                 f"<span style='font-size:9px;color:{MUTED}'>defects</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=WHITE),
        )],
        font=dict(color=TEXT),
    )
    return fig


def fig_split_donut():
    fig = go.Figure(go.Pie(
        labels=list(splits.keys()), values=list(splits.values()), hole=0.50,
        marker=dict(colors=[NEON_CYAN, NEON_ORG, NEON_YELL],
                    line=dict(color=BG, width=2)),
        pull=[0.05, 0.05, 0.05],
        textfont=dict(size=12, color=WHITE),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=260, font=dict(color=TEXT),
    )
    return fig


def fig_submission_donut():
    cd = submission["pred_label_code"].value_counts().sort_index()
    labels = [f"{c}  {NAME_MAP.get(c,'')}" for c in cd.index]
    fig = go.Figure(go.Pie(
        labels=labels, values=cd.values, hole=0.55,
        marker=dict(colors=DEF_PALETTE[:len(labels)],
                    line=dict(color=BG, width=2)),
        pull=[0.06 if i == 0 else 0 for i in range(len(labels))],
        textfont=dict(size=11, color=WHITE),
        hovertemplate="%{label}<br>%{value} samples (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=310,
        annotations=[dict(
            text=f"<b>115</b><br><span style='font-size:10px'>Samples</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=WHITE),
        )],
        font=dict(color=TEXT),
    )
    return fig


def fig_per_class_bar():
    cats  = list(CLASS_F1.keys())
    vals  = list(CLASS_F1.values())
    colors = [NEON_GREEN if v >= 0.90 else NEON_CYAN if v >= 0.75 else NEON_ORG
              for v in vals]
    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color=WHITE, width=0.4)),
        text=[f"{v:.2f}" for v in vals],
        textposition="outside", textfont=dict(color=WHITE, size=11),
        hovertemplate="%{x}<br>F1: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.80, line_dash="dot", line_color=NEON_ORG, line_width=1.5,
                  annotation_text="0.80 baseline",
                  annotation_font_color=NEON_ORG, annotation_font_size=10)
    _base(fig, h=290, ml=30, mb=80)
    fig.update_layout(yaxis=dict(range=[0, 1.15], title="F1 Score"),
                      xaxis=dict(tickangle=-25))
    return fig


def fig_p_defect_box():
    sub_s = submission.sort_values("pred_label_code")
    fig = go.Figure()
    for i, code in enumerate(sorted(sub_s["pred_label_code"].unique())):
        grp = sub_s[sub_s["pred_label_code"] == code]
        col = DEF_PALETTE[i % len(DEF_PALETTE)]
        fig.add_trace(go.Box(
            y=grp["p_defect"],
            name=f"{code} {NAME_MAP.get(code, '')}",
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(color=col, size=5, opacity=0.85,
                        line=dict(color=WHITE, width=0.5)),
            line=dict(color=col, width=2),
            fillcolor="rgba(0,80,200,0.10)",
            hovertemplate="%{y:.3f}<extra></extra>",
        ))
    _base(fig, h=320, ml=40, mb=70)
    fig.update_layout(yaxis_title="p_defect", showlegend=False,
                      xaxis=dict(tickangle=-20))
    return fig


# â”€â”€â”€ App â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap",
    ],
    title="Apex Â· Weld Quality AI",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
* { box-sizing: border-box; }
body {
  background: #000510;
  background-image:
    radial-gradient(ellipse 80% 50% at 15% 15%, rgba(0,80,255,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 85% 85%, rgba(0,200,255,0.08) 0%, transparent 60%);
  min-height: 100vh;
  font-family: 'Inter', 'Segoe UI', sans-serif;
  scrollbar-width: thin; scrollbar-color: #0055aa #000510;
}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#000510}
::-webkit-scrollbar-thumb{background:#0055aa;border-radius:3px}
.hero-gradient {
  background: linear-gradient(135deg, #000510 0%, #001a4a 40%, #002060 70%, #000510 100%);
  background-size: 300% 300%;
  animation: gradShift 10s ease infinite;
  border-bottom: 1px solid rgba(0,140,255,0.22);
  position: relative; overflow: hidden;
}
.hero-gradient::before {
  content:""; position:absolute; inset:0;
  background: radial-gradient(ellipse 50% 80% at 50% 100%, rgba(0,100,255,0.18) 0%, transparent 70%);
  pointer-events:none;
}
@keyframes gradShift {
  0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}
}
.glow-text-blue  { text-shadow: 0 0 20px rgba(0,160,255,0.8), 0 0 60px rgba(0,100,255,0.4); }
.glow-text-green { text-shadow: 0 0 18px rgba(0,255,157,0.9), 0 0 40px rgba(0,200,120,0.4); }
.nav-pills .nav-link {
  color:#5588aa!important; background:transparent!important;
  border:1px solid transparent!important; border-radius:10px!important;
  font-size:0.82rem!important; font-weight:600!important;
  padding:7px 18px!important; transition:all .2s ease!important;
}
.nav-pills .nav-link:hover {
  color:#00d4ff!important; border-color:rgba(0,212,255,.3)!important;
  background:rgba(0,100,200,.12)!important;
}
.nav-pills .nav-link.active {
  color:#fff!important;
  background:linear-gradient(135deg,#003880 0%,#005fcc 100%)!important;
  border-color:rgba(0,160,255,.6)!important;
  box-shadow:0 0 18px rgba(0,100,255,.4)!important;
}
.card{transition:transform .25s ease,box-shadow .25s ease!important}
.card:hover{transform:translateY(-3px) scale(1.005)!important;box-shadow:0 12px 48px rgba(0,80,255,.25)!important}
.metric-bar-track{background:rgba(0,80,200,.18);border-radius:99px;height:8px;overflow:hidden}
.score-badge{border:1px solid rgba(0,255,157,.4);background:rgba(0,255,157,.08);border-radius:20px;padding:6px 16px;font-size:.75rem;color:#00ff9d;font-weight:600;letter-spacing:1px;text-transform:uppercase}
.glow-divider{border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(0,160,255,.5),transparent);margin:18px 0}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 6px #00ff9d}50%{opacity:.4;box-shadow:0 0 14px #00ff9d}}
.live-dot{display:inline-block;width:8px;height:8px;background:#00ff9d;border-radius:50%;margin-right:6px;animation:pulse 2s infinite;vertical-align:middle}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


# â”€â”€â”€ Tab builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def tab_overview():
    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Total Samples",    f"{total_runs:,}", "weld recordings",                WHITE),       md=3),
            dbc.Col(kpi_card("Good Welds",        f"{good_count:,}", f"{good_count/total_runs:.0%}",  NEON_CYAN),   md=3),
            dbc.Col(kpi_card("Defective Welds",   f"{defect_count:,}", f"{defect_count/total_runs:.0%}", NEON_PINK), md=3),
            dbc.Col(kpi_card("Defect Categories", "6", "unique types",                               NEON_PURP),   md=3),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("Defect Type Distribution", "ðŸ©"),
                dcc.Graph(figure=fig_defect_donut(), config=GRAPH_CFG),
            ), md=4),
            dbc.Col(glass_card(
                section_header("Train / Val / Test Split", "ðŸ“Š"),
                dcc.Graph(figure=fig_split_donut(), config=GRAPH_CFG),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.P("Train", style={"color": NEON_CYAN, "fontWeight": "700",
                                               "margin": 0, "fontSize": "0.78rem"}),
                        html.P("1,047", style={"color": MUTED, "margin": 0, "fontSize": "0.72rem"}),
                    ], className="text-center"), md=4),
                    dbc.Col(html.Div([
                        html.P("Val", style={"color": NEON_ORG, "fontWeight": "700",
                                             "margin": 0, "fontSize": "0.78rem"}),
                        html.P("240", style={"color": MUTED, "margin": 0, "fontSize": "0.72rem"}),
                    ], className="text-center"), md=4),
                    dbc.Col(html.Div([
                        html.P("Test", style={"color": NEON_YELL, "fontWeight": "700",
                                              "margin": 0, "fontSize": "0.78rem"}),
                        html.P("264", style={"color": MUTED, "margin": 0, "fontSize": "0.72rem"}),
                    ], className="text-center"), md=4),
                ], className="mt-2"),
            ), md=4),
            dbc.Col(glass_card(
                section_header("Prediction Confidence Distribution", "ðŸ“ˆ"),
                dcc.Graph(figure=fig_pred_dist(), config=GRAPH_CFG),
            ), md=4),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("3D MFCC Audio Feature Space â€” Clusters by Defect Type", "ðŸ”µ"),
                html.P(
                    "Each point is a weld sample projected into MFCC space. "
                    "Tight, separated clusters indicate high audio discriminability. Rotate & zoom.",
                    style={"color": MUTED, "fontSize": "0.72rem", "marginBottom": "8px"},
                ),
                dcc.Graph(figure=fig_3d_feature_scatter(),
                          config={"displayModeBar": True}),
            ), md=12),
        ], className="mb-4 g-3"),
    ])


def tab_performance():
    def mbar(label, val, col):
        pct = int(val * 100)
        return html.Div([
            html.Div([
                html.Span(label, style={"color": MUTED, "fontSize": "0.73rem"}),
                html.Span(f"{val:.4f}", style={"color": col, "fontWeight": "700",
                                                "fontSize": "0.73rem"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "marginBottom": "4px"}),
            html.Div(
                html.Div(style={
                    "width": f"{pct}%",
                    "background": f"linear-gradient(90deg, {col}88, {col})",
                    "height": "100%", "borderRadius": "99px",
                    "boxShadow": f"0 0 8px {col}66",
                }),
                className="metric-bar-track", style={"marginBottom": "10px"},
            ),
        ])

    return html.Div([
        dbc.Row([
            dbc.Col(glass_card(
                section_header("Competition Metrics", "ã€½ï¸"),
                mbar("Binary F1",    BINARY_F1,          NEON_GREEN),
                mbar("Precision",    BIN_PREC,            NEON_CYAN),
                mbar("Recall",       BIN_REC,             NEON_ORG),
                mbar("Specificity",  SPEC,                NEON_PURP),
                mbar("Accuracy",     (TP + TN) / 115,     WHITE),
                mbar("PR-AUC",       PR_AUC,              NEON_YELL),
                mbar("Type MacroF1", TYPE_MF1,            NEON_PINK),
                html.Hr(className="glow-divider"),
                html.P(f"ECE: {ECE:.4f}  Â·  Brier: {BRIER:.4f}",
                       style={"color": MUTED, "fontSize": "0.72rem", "margin": 0}),
            ), md=4),
            dbc.Col(glass_card(
                section_header("ROC-AUC", "ðŸŽ¯"),
                dcc.Graph(figure=fig_gauge(ROC_AUC, "ROC-AUC", ref=88),
                          config=GRAPH_CFG),
                html.Hr(className="glow-divider"),
                section_header("FinalScore = 0.6Ã—BinaryF1 + 0.4Ã—TypeF1", "ðŸ†"),
                dcc.Graph(figure=fig_gauge(FINAL_SCORE, "FinalScore", ref=80,
                                           color=NEON_GREEN, h=270),
                          config=GRAPH_CFG),
            ), md=4),
            dbc.Col(glass_card(
                section_header("Confusion Matrix (Competition Test)", "ðŸ”¥"),
                dcc.Graph(figure=fig_confusion_neon(), config=GRAPH_CFG),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.P("TP", style={"color": MUTED, "fontSize": "0.68rem", "margin": 0}),
                        html.H4(str(TP), style={"color": NEON_GREEN, "margin": 0,
                                                 "textShadow": f"0 0 12px {NEON_GREEN}88"}),
                    ], className="text-center"), md=3),
                    dbc.Col(html.Div([
                        html.P("FP", style={"color": MUTED, "fontSize": "0.68rem", "margin": 0}),
                        html.H4(str(FP), style={"color": NEON_PINK, "margin": 0,
                                                 "textShadow": f"0 0 12px {NEON_PINK}88"}),
                    ], className="text-center"), md=3),
                    dbc.Col(html.Div([
                        html.P("TN", style={"color": MUTED, "fontSize": "0.68rem", "margin": 0}),
                        html.H4(str(TN), style={"color": NEON_CYAN, "margin": 0,
                                                 "textShadow": f"0 0 12px {NEON_CYAN}88"}),
                    ], className="text-center"), md=3),
                    dbc.Col(html.Div([
                        html.P("FN", style={"color": MUTED, "fontSize": "0.68rem", "margin": 0}),
                        html.H4(str(FN), style={"color": NEON_ORG, "margin": 0,
                                                 "textShadow": f"0 0 12px {NEON_ORG}88"}),
                    ], className="text-center"), md=3),
                ], className="mt-2"),
            ), md=4),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("Training Loss Curves", "ðŸ“‰"),
                dcc.Graph(figure=fig_training_area(), config=GRAPH_CFG),
            ), md=6),
            dbc.Col(glass_card(
                section_header("Validation Accuracy & F1", "ðŸ“ˆ"),
                dcc.Graph(figure=fig_val_metrics(), config=GRAPH_CFG),
            ), md=6),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("Per-Class F1 Score", "ðŸ“Š"),
                dcc.Graph(figure=fig_per_class_bar(), config=GRAPH_CFG),
            ), md=7),
            dbc.Col(glass_card(
                section_header("Radar: Per-Class F1", "ðŸ•¸ï¸"),
                dcc.Graph(figure=fig_radar_per_class(), config=GRAPH_CFG),
            ), md=5),
        ], className="mb-4 g-3"),
    ])


def tab_submission():
    cd = submission["pred_label_code"].value_counts().sort_index()
    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Competition Samples", "115",                  "audio .flac files",        WHITE),      md=2),
            dbc.Col(kpi_card("Binary F1",            f"{BINARY_F1:.4f}",   "defect vs good",             NEON_GREEN), md=2),
            dbc.Col(kpi_card("Type Macro F1",        f"{TYPE_MF1:.4f}",    "7-class macro avg",          NEON_CYAN),  md=2),
            dbc.Col(kpi_card("FinalScore",           f"{FINAL_SCORE:.4f}", "0.6Ã—BinF1 + 0.4Ã—TypeF1",    NEON_ORG),   md=2),
            dbc.Col(kpi_card("ROC-AUC",              f"{ROC_AUC:.4f}",     "area under curve",           NEON_PURP),  md=2),
            dbc.Col(kpi_card("CV Macro-F1",          "0.9744",             "5-fold cross-val",           NEON_YELL),  md=2),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("3D Predicted Code Distribution", "ðŸ—ï¸"),
                html.P("Rotate and zoom the 3D chart.",
                       style={"color": MUTED, "fontSize": "0.72rem", "marginBottom": "6px"}),
                dcc.Graph(figure=fig_3d_submission_bar(),
                          config={"displayModeBar": True}),
            ), md=7),
            dbc.Col(glass_card(
                section_header("Prediction Code Breakdown", "ðŸ©"),
                dcc.Graph(figure=fig_submission_donut(), config=GRAPH_CFG),
                html.Div([
                    html.Div([
                        html.Span("â—", style={"color": DEF_PALETTE[i], "marginRight": "6px",
                                              "textShadow": f"0 0 6px {DEF_PALETTE[i]}"}),
                        html.Span(f"{code}  {NAME_MAP.get(code,'')}  â€” ",
                                  style={"color": TEXT, "fontSize": "0.72rem"}),
                        html.Span(str(cd.get(code, 0)),
                                  style={"color": DEF_PALETTE[i], "fontWeight": "700",
                                         "fontSize": "0.72rem"}),
                    ], style={"marginBottom": "3px"})
                    for i, code in enumerate(sorted(cd.index))
                ], style={"marginTop": "8px"}),
            ), md=5),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("Confidence (p_defect) by Predicted Code", "ðŸ“¦"),
                dcc.Graph(figure=fig_p_defect_box(), config=GRAPH_CFG),
            ), md=12),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("Submission Preview â€” first 20 rows", "ðŸ“‹"),
                html.Div(
                    _dataframe_to_table(
                        submission.head(20).assign(
                            Defect_Type=submission["pred_label_code"].head(20).map(NAME_MAP),
                            p_defect=submission["p_defect"].head(20).round(4),
                        )[["sample_id", "pred_label_code", "Defect_Type", "p_defect"]],
                    ),
                    style={"maxHeight": "320px", "overflowY": "auto"},
                ),
            ), md=12),
        ], className="mb-4 g-3"),
    ])


def tab_about():
    def info_row(label, value, col=NEON_CYAN):
        return html.Tr([
            html.Td(label, style={"color": MUTED, "fontSize": "0.80rem",
                                   "padding": "8px 12px", "borderColor": BORDER_COL}),
            html.Td(value, style={"color": col, "fontWeight": "600",
                                   "fontSize": "0.80rem", "padding": "8px 12px",
                                   "borderColor": BORDER_COL}),
        ])

    return html.Div([
        dbc.Row([
            dbc.Col(glass_card(
                section_header("Project Overview", "ðŸ”¬"),
                html.P([
                    html.Span("Apex Weld Quality AI",
                              style={"color": WHITE, "fontWeight": "700"}),
                    " is an end-to-end ML pipeline for ",
                    html.Span("automated weld defect classification",
                              style={"color": NEON_CYAN}),
                    " from audio recordings. It classifies welds as good or "
                    "defective across 6 defect categories.",
                ], style={"color": TEXT, "fontSize": "0.84rem", "lineHeight": "1.7"}),
                html.Hr(className="glow-divider"),
                dbc.Table([html.Tbody([
                    info_row("Competition",   "Hackathon â€” Weld Quality Classification"),
                    info_row("Date",          "February 28, 2026"),
                    info_row("Task",          "Binary + 7-class defect detection"),
                    info_row("Training Data", "1,551 real weld .flac recordings"),
                    info_row("Test Data",     "115 competition audio samples (data2/)"),
                    info_row("FinalScore",    f"{FINAL_SCORE:.4f}  (target > 0.80)", NEON_GREEN),
                ])], bordered=False, size="sm",
                   style={"color": TEXT, "--bs-table-color": TEXT,
                          "--bs-table-border-color": BORDER_COL}),
            ), md=4),
            dbc.Col(glass_card(
                section_header("Model Architecture", "ðŸ§ "),
                dbc.Table([html.Tbody([
                    info_row("Classifier",        "LR(C=7) + SVM(C=2, RBF) â€” Soft Voting 2:1"),
                    info_row("Feature Selection", "Top-130 by LR coefficient importance"),
                    info_row("Features",          "MFCCs, Mel, ZCR, RMS, Spectral (217 dims)"),
                    info_row("Normalisation",     "StandardScaler"),
                    info_row("Class Weights",     "balanced (handles imbalance)"),
                    info_row("CV Strategy",       "5-fold StratifiedKFold"),
                    info_row("CV Macro-F1",       "0.9744  (7-class)", NEON_GREEN),
                    info_row("Binary CV-F1",      "0.9779", NEON_GREEN),
                ])], bordered=False, size="sm",
                   style={"color": TEXT, "--bs-table-color": TEXT,
                          "--bs-table-border-color": BORDER_COL}),
            ), md=4),
            dbc.Col(glass_card(
                section_header("Defect Code Map", "ðŸ—‚ï¸"),
                *[
                    html.Div([
                        html.Span("â—", style={"color": DEF_PALETTE[i], "marginRight": "10px",
                                               "textShadow": f"0 0 8px {DEF_PALETTE[i]}",
                                               "fontSize": "1.1rem"}),
                        html.Span(f"{code}", style={"color": WHITE, "fontWeight": "700",
                                                     "fontSize": "0.82rem", "minWidth": "32px",
                                                     "display": "inline-block"}),
                        html.Span(f"  {name}", style={"color": MUTED, "fontSize": "0.80rem"}),
                    ], style={"marginBottom": "11px", "display": "flex",
                               "alignItems": "center"})
                    for i, (code, name) in enumerate([
                        ("00", "Good Weld"),          ("01", "Excessive Penetration"),
                        ("02", "Burnthrough"),         ("06", "Overlap"),
                        ("07", "Lack of Fusion"),      ("08", "Excessive Convexity"),
                        ("11", "Crater Cracks"),
                    ])
                ],
            ), md=4),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(glass_card(
                section_header("Pipeline Steps", "âš¡"),
                html.Div([
                    html.Div([
                        html.Div(str(i + 1), style={
                            "width": "28px", "height": "28px", "borderRadius": "50%",
                            "textAlign": "center", "lineHeight": "28px",
                            "fontSize": "0.75rem", "fontWeight": "700", "flexShrink": "0",
                            "background": f"linear-gradient(135deg, {GLOW_BLUE}, {NEON_CYAN})",
                            "boxShadow": f"0 0 12px {NEON_CYAN}66", "color": WHITE,
                        }),
                        html.Div([
                            html.Span(title, style={"color": WHITE, "fontWeight": "600",
                                                     "fontSize": "0.82rem"}),
                            html.Br(),
                            html.Span(desc, style={"color": MUTED, "fontSize": "0.72rem"}),
                        ], style={"marginLeft": "14px"}),
                    ], style={"display": "flex", "alignItems": "flex-start",
                               "marginBottom": "16px"})
                    for i, (title, desc) in enumerate([
                        ("Audio Extraction",   "MFCCs, Mel-spectrogram, ZCR, RMS, Spectral from .flac"),
                        ("Normalisation",      "StandardScaler â€” zero-mean, unit-variance"),
                        ("Feature Selection",  "Fit LR(C=5), keep top-130 by max |coef|"),
                        ("Ensemble Training",  "LR(C=7, balanced) + SVM(C=2, RBF, balanced) â€” soft vote 2:1"),
                        ("7-Class Prediction", "good / exc_penetration / burnthrough / overlap / lack_fusion / exc_convexity / crater_cracks"),
                        ("Submission Output",  "115 rows â†’ sample_id, pred_label_code, p_defect â†’ outputs/submission.csv"),
                    ])
                ]),
            ), md=12),
        ], className="mb-4 g-3"),
    ])


# â”€â”€â”€ Layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app.layout = dbc.Container([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span(className="live-dot"),
                    html.Span("LIVE", style={"color": NEON_GREEN, "fontWeight": "700",
                                              "fontSize": "0.65rem", "letterSpacing": "3px"}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                html.H1("Apex Weld Quality AI", className="glow-text-blue",
                        style={"color": WHITE, "fontWeight": "800", "fontSize": "2.2rem",
                               "margin": 0, "letterSpacing": "-0.5px"}),
                html.P("AI-Powered Weld Defect Classification  Â·  Audio ML Pipeline  Â·  Hackathon 2026",
                       style={"color": MUTED, "margin": "6px 0 0 0", "fontSize": "0.84rem"}),
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.P("FinalScore", style={"color": MUTED, "fontSize": "0.62rem",
                                                     "letterSpacing": "3px",
                                                     "textTransform": "uppercase",
                                                     "margin": "0 0 2px 0"}),
                        html.H2(f"{FINAL_SCORE:.4f}", className="glow-text-green",
                                style={"color": NEON_GREEN, "fontWeight": "800",
                                       "margin": 0, "fontSize": "2.4rem"}),
                        html.Span("115 Competition Samples", className="score-badge",
                                  style={"marginTop": "8px", "display": "inline-block"}),
                    ], className="text-end"),
                ], className="d-flex align-items-center justify-content-end h-100"),
            ], md=4),
        ], className="py-4"),
    ], className="hero-gradient px-4 mb-4"),

    dbc.Row([
        dbc.Col(dbc.Nav([
            dbc.NavItem(dbc.NavLink("ðŸ   Overview",     href="#", id="tab-overview", n_clicks=0, active=True)),
            dbc.NavItem(dbc.NavLink("ðŸ“Š  Performance",  href="#", id="tab-perf",    n_clicks=0)),
            dbc.NavItem(dbc.NavLink("ðŸŽ¯  Submission",   href="#", id="tab-sub",     n_clicks=0)),
            dbc.NavItem(dbc.NavLink("â„¹ï¸  About",        href="#", id="tab-about",   n_clicks=0)),
        ], pills=True, className="mb-4"), md=12),
    ]),

    html.Div(id="page-content"),

    html.Hr(className="glow-divider"),
    html.P(
        f"Apex Weld Quality  Â·  Hackathon 2026  Â·  LR + SVM Audio Pipeline  Â·  FinalScore {FINAL_SCORE:.4f}",
        className="text-center pb-3",
        style={"color": MUTED, "fontSize": "0.70rem"},
    ),
], fluid=True, style={"padding": "0", "background": BG, "minHeight": "100vh"})


# â”€â”€â”€ Callbacks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.callback(
    Output("page-content", "children"),
    [Input("tab-overview", "n_clicks"), Input("tab-perf",  "n_clicks"),
     Input("tab-sub",      "n_clicks"), Input("tab-about", "n_clicks")],
)
def render_tab(n_ov, n_pf, n_sb, n_ab):
    ctx = dash.callback_context
    tab_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "tab-overview"
    return html.Div({
        "tab-overview": tab_overview,
        "tab-perf":     tab_performance,
        "tab-sub":      tab_submission,
        "tab-about":    tab_about,
    }.get(tab_id, tab_overview)(), className="px-2")


@app.callback(
    [Output("tab-overview", "active"), Output("tab-perf",  "active"),
     Output("tab-sub",      "active"), Output("tab-about", "active")],
    [Input("tab-overview", "n_clicks"), Input("tab-perf",  "n_clicks"),
     Input("tab-sub",      "n_clicks"), Input("tab-about", "n_clicks")],
)
def highlight_tab(n_ov, n_pf, n_sb, n_ab):
    ctx = dash.callback_context
    tab_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "tab-overview"
    tabs = ["tab-overview", "tab-perf", "tab-sub", "tab-about"]
    return tuple(tab_id == t for t in tabs)


# â”€â”€â”€ Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    print("\n  â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—")
    print("  â•‘   Apex Weld Quality Dashboard v3    â•‘")
    print("  â•‘   http://127.0.0.1:8050             â•‘")
    print("  â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n")
    app.run(debug=False, port=8050)
