"""Model evaluation and confusion matrix analytics for DA2 cybersecurity models."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEVERITY_CLASSES = ["Low", "Medium", "High"]
CYAN = "#35D0E8"
BLUE = "#4F8CFF"
MUTED = "#8FA3BD"
NAVY_BG = "#08111F"
NAVY_BORDER = "#20324B"

def load_test_predictions(out_dir):
    """
    Loads genuine held-out test predictions artifact.
    Returns DataFrame or None if missing.
    """
    path = Path(out_dir) / "tuned_test_predictions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        required = {"severity_class", ".pred_class", "model"}
        if not required.issubset(df.columns):
            return None
        return df
    except Exception:
        return None

def compute_confusion_matrix(df_model, normalize=False):
    """
    Computes confusion matrix with exact order [Low, Medium, High].
    Rows: Actual Severity, Columns: Predicted Severity.
    """
    y_true = df_model["severity_class"].astype(str)
    y_pred = df_model[".pred_class"].astype(str)
    
    cm = pd.crosstab(y_true, y_pred).reindex(
        index=SEVERITY_CLASSES,
        columns=SEVERITY_CLASSES,
        fill_value=0
    )
    cm.index.name = "Actual Severity"
    cm.columns.name = "Predicted Severity"
    
    if normalize:
        row_sums = cm.sum(axis=1)
        # Avoid division by zero
        row_sums_safe = row_sums.replace(0, 1)
        cm_norm = cm.div(row_sums_safe, axis=0) * 100.0
        return cm, cm_norm
    return cm, None

def compute_detailed_metrics(df_model):
    """
    Computes exact per-class precision, recall, F1, support, and macro-averages.
    """
    cm, _ = compute_confusion_matrix(df_model, normalize=False)
    total_samples = int(cm.values.sum())
    diag = np.diag(cm.values)
    accuracy = float(diag.sum() / total_samples) if total_samples > 0 else 0.0
    
    per_class = []
    for idx, cls_name in enumerate(SEVERITY_CLASSES):
        tp = int(cm.iloc[idx, idx])
        fn = int(cm.iloc[idx, :].sum() - tp)
        fp = int(cm.iloc[:, idx].sum() - tp)
        tn = int(total_samples - (tp + fn + fp))
        support = int(tp + fn)
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        per_class.append({
            "Severity Class": cls_name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support (Actual)": support,
            "True Positives": tp,
            "False Positives": fp,
            "False Negatives": fn,
            "True Negatives": tn,
        })
    
    metrics_df = pd.DataFrame(per_class)
    
    macro_prec = float(metrics_df["Precision"].mean())
    macro_rec = float(metrics_df["Recall"].mean())
    macro_f1 = float(metrics_df["F1-Score"].mean())
    
    # Weighted F1
    weighted_f1 = float((metrics_df["F1-Score"] * metrics_df["Support (Actual)"]).sum() / total_samples) if total_samples > 0 else 0.0
    
    summary = {
        "Accuracy": accuracy,
        "Macro Precision": macro_prec,
        "Macro Recall": macro_rec,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Total Test Records": total_samples
    }
    return metrics_df, summary

def analyze_error_risks(df_model):
    """
    Analyzes specific cybersecurity operational risk scenarios from misclassifications.
    """
    cm, _ = compute_confusion_matrix(df_model, normalize=False)
    
    # Actual High predicted as Low or Medium (Critical Failure)
    high_as_low = int(cm.loc["High", "Low"]) if "High" in cm.index and "Low" in cm.columns else 0
    high_as_med = int(cm.loc["High", "Medium"]) if "High" in cm.index and "Medium" in cm.columns else 0
    total_high_missed = high_as_low + high_as_med
    actual_high_total = int(cm.loc["High"].sum()) if "High" in cm.index else 0
    high_detection_rate = float(cm.loc["High", "High"] / actual_high_total * 100.0) if actual_high_total > 0 else 0.0
    
    # Actual Low predicted as High (False Alarm)
    low_as_high = int(cm.loc["Low", "High"]) if "Low" in cm.index and "High" in cm.columns else 0
    med_as_high = int(cm.loc["Medium", "High"]) if "Medium" in cm.index and "High" in cm.columns else 0
    total_false_highs = low_as_high + med_as_high
    
    # Cross errors between Low and Medium
    med_as_low = int(cm.loc["Medium", "Low"]) if "Medium" in cm.index and "Low" in cm.columns else 0
    low_as_med = int(cm.loc["Low", "Medium"]) if "Low" in cm.index and "Medium" in cm.columns else 0
    
    return {
        "actual_high_total": actual_high_total,
        "high_correct": int(cm.loc["High", "High"]) if "High" in cm.index else 0,
        "high_detection_rate": high_detection_rate,
        "high_as_low": high_as_low,
        "high_as_med": high_as_med,
        "total_high_missed": total_high_missed,
        "low_as_high": low_as_high,
        "med_as_high": med_as_high,
        "total_false_highs": total_false_highs,
        "med_as_low": med_as_low,
        "low_as_med": low_as_med,
    }

def build_confusion_matrix_figure(cm_raw, cm_norm=None, show_normalized=False, model_name=""):
    """
    Renders an interactive confusion matrix heatmap with clear labels and tooltips.
    """
    display_matrix = cm_norm if show_normalized and cm_norm is not None else cm_raw
    text_annotations = []
    
    for row_idx, actual_cls in enumerate(SEVERITY_CLASSES):
        row_ann = []
        for col_idx, pred_cls in enumerate(SEVERITY_CLASSES):
            raw_val = int(cm_raw.loc[actual_cls, pred_cls])
            norm_val = float(cm_norm.loc[actual_cls, pred_cls]) if cm_norm is not None else 0.0
            
            if show_normalized:
                ann_str = f"<b>{norm_val:.1f}%</b><br><span style='font-size:10px;color:#8FA3BD;'>({raw_val:,} events)</span>"
            else:
                ann_str = f"<b>{raw_val:,}</b><br><span style='font-size:10px;color:#8FA3BD;'>({norm_val:.1f}% of actual)</span>"
            row_ann.append(ann_str)
        text_annotations.append(row_ann)
    
    colorscale = [
        [0.0, "#081326"],
        [0.2, "#0D2547"],
        [0.5, "#184A7E"],
        [0.8, "#2878B5"],
        [1.0, CYAN]
    ]
    
    fig = go.Figure(
        data=go.Heatmap(
            z=display_matrix.values,
            x=SEVERITY_CLASSES,
            y=SEVERITY_CLASSES,
            text=text_annotations,
            texttemplate="%{text}",
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text="Recall %" if show_normalized else "Incident Count", font=dict(color=CYAN, size=11)),
                tickfont=dict(color="#E7EEF8", size=10),
                thickness=12,
                len=0.75,
                bgcolor="rgba(11,23,40,0.8)",
                bordercolor=NAVY_BORDER,
                borderwidth=1
            ),
            hovertemplate=(
                "<b>Actual Severity: %{y}</b><br>"
                "<b>Predicted Severity: %{x}</b><br>"
                "Incidents: %{customdata[0]:,}<br>"
                "Row Percentage: %{customdata[1]:.2f}% of Actual %{y}<extra></extra>"
            ),
            customdata=np.dstack((cm_raw.values, cm_norm.values if cm_norm is not None else np.zeros_like(cm_raw.values))),
        )
    )
    
    title_suffix = " (Row-Normalized / Recall %)" if show_normalized else " (Raw Incident Counts)"
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix: {model_name}{title_suffix}",
            font=dict(color="#E7EEF8", size=14, family="Inter, sans-serif")
        ),
        xaxis=dict(
            title=dict(text="Predicted Severity Class", font=dict(color="#E7EEF8", size=13)),
            tickfont=dict(color="#E7EEF8", size=11),
            side="bottom",
            gridcolor=NAVY_BORDER
        ),
        yaxis=dict(
            title=dict(text="Actual Severity Class", font=dict(color="#E7EEF8", size=13)),
            tickfont=dict(color="#E7EEF8", size=11),
            autorange="reversed",  # Keeps Low, Medium, High from top to bottom
            gridcolor=NAVY_BORDER
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=50, b=50),
        height=380,
    )
    return fig
