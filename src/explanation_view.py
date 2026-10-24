"""Visualization and diagnostic helpers for local prediction explanations."""
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CYAN = "#35D0E8"
BLUE = "#4F8CFF"
MUTED = "#8FA3BD"
NAVY_BORDER = "#20324B"

def clean_feature_name(raw_name):
    """Translates model-baked variable identifiers into readable analyst terms."""
    if not isinstance(raw_name, str):
        return str(raw_name)
    s = raw_name
    s = s.replace(".", " ").replace("_", " ")
    
    # Common prefixes
    prefixes = {
        "incident type ": "Attack Vector: ",
        "initiator country ": "Attributed Origin: ",
        "receiver category ": "Victim Sector: ",
        "target ": "Target Category: ",
        "is ": "Indicator: ",
        "has ": "Impact: ",
        "mitre impact ": "MITRE Impact: ",
        "mitre initial access ": "MITRE Initial Access: ",
        "state responsibility actor ": "State Responsibility: ",
        "initiator category ": "Actor Category: ",
        "offline conflict issue ": "Conflict Issue: ",
    }
    for pfx, label in prefixes.items():
        if s.lower().startswith(pfx):
            return label + s[len(pfx):].title()
    return s.title()

def build_local_explanation_figure(exp_df, predicted_class, method_name):
    """
    Renders an interactive horizontal bar chart showing top feature contributions
    pushing toward or against the predicted severity class.
    """
    if exp_df.empty or "contribution" not in exp_df.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No feature contributions calculated", showarrow=False, font=dict(color=MUTED, size=14))]
        )
        return fig

    plot_df = exp_df.copy().head(12)
    plot_df["display_name"] = plot_df["feature"].apply(clean_feature_name)
    
    # Sort for bottom-up horizontal bar display
    plot_df = plot_df.iloc[::-1]

    colors = []
    for val in plot_df["contribution"]:
        if val > 0:
            colors.append("#35D0E8" if predicted_class == "Low" else ("#F6B94A" if predicted_class == "Medium" else "#FF647C"))
        else:
            colors.append("#4F8CFF")

    x_title = "Shapley Value Contribution (Log-Odds)" if "TreeSHAP" in method_name else (
        "Marginal Probability Impact (ΔP)" if "Sensitivity" in method_name or "Perturbation" in method_name else "Relative Importance / Split Threshold"
    )

    fig = go.Figure(
        data=go.Bar(
            x=plot_df["contribution"],
            y=plot_df["display_name"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=NAVY_BORDER, width=1)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Local Contribution: <b>%{x:+.4f}</b><br>"
                "Direction: %{customdata[0]}<br>"
                "Method: %{customdata[1]}<extra></extra>"
            ),
            customdata=plot_df[["direction", "method"]],
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Top Influential Features for {predicted_class} Severity Prediction",
            font=dict(color="#E7EEF8", size=13, family="Inter, sans-serif")
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="#E7EEF8", size=11)),
            tickfont=dict(color="#E7EEF8", size=10),
            gridcolor=NAVY_BORDER,
            zerolinecolor="#506680",
            zerolinewidth=1.5
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#E7EEF8", size=10),
            gridcolor="rgba(0,0,0,0)"
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=35, b=25),
        height=max(320, len(plot_df) * 28),
    )
    return fig
