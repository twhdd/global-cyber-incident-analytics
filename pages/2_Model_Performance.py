from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation import (
    load_test_predictions,
    compute_confusion_matrix,
    compute_detailed_metrics,
    analyze_error_risks,
    build_confusion_matrix_figure,
    SEVERITY_CLASSES
)

st.set_page_config(page_title="Model Performance Lab", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "r_outputs"

st.markdown("""
<style>
.stApp { background:#08111F; color:#E7EEF8; font-family:Inter,sans-serif; }
[data-testid="stSidebar"] { background:#0B1728; border-right:1px solid #20324B; }
.eyebrow { color:#35D0E8; font-family:monospace; letter-spacing:.15em; font-size:.72rem; }
.metric-card { background:linear-gradient(135deg,#12233A,#0E1A2B); border:1px solid #263A55; border-radius:12px; padding:0.9rem; }
.metric-card .val { font-size:1.6rem; font-weight:700; color:#E7EEF8; margin-top:0.2rem; }
.metric-card .lbl { font-size:0.75rem; text-transform:uppercase; color:#8FA3BD; letter-spacing:0.06em; }
.metric-card .sub { font-size:0.72rem; color:#35D0E8; margin-top:0.15rem; }
.risk-box { border-radius:10px; padding:0.9rem; margin-bottom:0.7rem; border-left:4px solid; }
.risk-critical { background:#1F1318; border-color:#FF647C; color:#FFB4C0; }
.risk-warning { background:#1C1912; border-color:#F6B94A; color:#FFE5B4; }
.risk-info { background:#0E1B2C; border-color:#35D0E8; color:#D3EDFB; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌐 CYBER INTEL")
    st.caption("GLOBAL INCIDENT MONITOR")
    st.page_link("app.py", label="Overview", icon=":material/dashboard:")
    st.page_link("pages/1_Incident_Explorer.py", label="Incident Explorer", icon=":material/search:")
    st.page_link("pages/2_Model_Performance.py", label="Model Performance Lab", icon=":material/monitoring:")
    st.page_link("pages/3_Feature_Insights.py", label="Feature Insights", icon=":material/insights:")
    st.page_link("pages/4_Severity_Prediction.py", label="Severity Prediction Demo", icon=":material/science:")

st.markdown('<div class="eyebrow">DA2 / MODEL EVALUATION & ERROR ANALYSIS</div>', unsafe_allow_html=True)
st.title("Model Performance Lab")
st.caption("Comprehensive evaluation of cross-validation results, held-out test metrics, confusion matrices, and error analysis.")

def load_csv(filename):
    path = OUT / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None

cv = load_csv("cv_12_model_results.csv")
test = load_csv("tuned_test_results.csv")
params = load_csv("best_hyperparameters.csv")
test_predictions = load_test_predictions(OUT)

if cv is None or test is None or params is None:
    st.error("One or more required R result CSVs are missing from data/r_outputs.")
    st.write("Expected files:")
    st.code("cv_12_model_results.csv\ntuned_test_results.csv\nbest_hyperparameters.csv")
    st.stop()

metric_labels = {
    "macro_f1": "Macro F1",
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "weighted_f1": "Weighted F1",
    "macro_roc_auc": "Macro ROC-AUC",
    "macro_pr_auc": "Macro PR-AUC",
}
available_metrics = [m for m in metric_labels if m in cv.columns and m in test.columns]

if "model" not in cv.columns or "model" not in test.columns:
    st.error("Expected a 'model' column in the R result files.")
    st.stop()

best_cv = cv.sort_values("macro_f1", ascending=False).iloc[0]
best_test = test.sort_values("macro_f1", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="metric-card"><div class="lbl">Models in CV Comparison</div><div class="val">{len(cv):,}</div><div class="sub">10-fold cross validation</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="lbl">Top CV Macro F1</div><div class="val">{best_cv["macro_f1"]:.4f}</div><div class="sub">{best_cv["model"]}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="lbl">Tuned Models Tested</div><div class="val">{len(test):,}</div><div class="sub">Held-out test split (20%)</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="lbl">Top Held-Out Macro F1</div><div class="val">{best_test["macro_f1"]:.4f}</div><div class="sub">{best_test["model"]}</div></div>', unsafe_allow_html=True)

tab_cm, tab_heldout, tab_cv, tab_params = st.tabs([
    "🎯 Confusion Matrix & Error Analysis",
    "📊 Held-Out Test Evaluation",
    "🔄 12-Model Cross-Validation",
    "⚙️ Tuned Hyperparameters"
])

# ==========================================
# TAB 1: CONFUSION MATRIX & ERROR ANALYSIS
# ==========================================
with tab_cm:
    st.subheader("Interactive Confusion Matrix & Error Diagnostics")
    st.caption(
        "Evaluate prediction matrices on the held-out test set (683 incidents, 20% stratified test split). "
        "Matrices use the canonical severity order: Low, Medium, High."
    )

    if test_predictions is None:
        st.warning(
            "Instance-level held-out test predictions (`data/r_outputs/tuned_test_predictions.csv`) were not found. "
            "To display the confusion matrix, ensure `tuned_test_predictions.csv` containing actual and predicted labels "
            "for the test set is located in `data/r_outputs/`."
        )
    else:
        available_models = sorted(test_predictions["model"].unique().tolist())
        default_model = "Extra Trees" if "Extra Trees" in available_models else available_models[0]
        
        col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])
        with col_ctrl1:
            selected_model = st.selectbox(
                "Select Model for Detailed Matrix",
                options=available_models,
                index=available_models.index(default_model)
            )
        with col_ctrl2:
            matrix_mode = st.radio(
                "Matrix View Mode",
                options=["Raw Incident Counts", "Row-Normalized Percentages (Recall %)"],
                index=0,
                horizontal=True
            )

        sub_preds = test_predictions[test_predictions["model"] == selected_model]
        cm_raw, cm_norm = compute_confusion_matrix(sub_preds, normalize=True)
        metrics_df, summary_stats = compute_detailed_metrics(sub_preds)
        error_risks = analyze_error_risks(sub_preds)

        show_norm = (matrix_mode == "Row-Normalized Percentages (Recall %)")
        
        # Confusion matrix visual
        cm_col, met_col = st.columns([1.3, 1.2])
        with cm_col:
            fig_cm = build_confusion_matrix_figure(cm_raw, cm_norm, show_normalized=show_norm, model_name=selected_model)
            st.plotly_chart(fig_cm, use_container_width=True)

        with met_col:
            st.markdown(f"**Performance Summary — {selected_model}**")
            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric("Test Accuracy", f"{summary_stats['Accuracy']:.2%}")
            with p2:
                st.metric("Macro F1", f"{summary_stats['Macro F1']:.4f}")
            with p3:
                st.metric("Weighted F1", f"{summary_stats['Weighted F1']:.4f}")

            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Per-Class Precision, Recall, and F1 Breakdown**")
            st.dataframe(
                metrics_df[["Severity Class", "Precision", "Recall", "F1-Score", "Support (Actual)"]].style.format({
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.4f}",
                    "Support (Actual)": "{:,}"
                }),
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # Cybersecurity Error Analysis & Risk Profiles
        st.subheader("Cybersecurity Risk Analysis: Critical Errors & Asymmetric Impact")
        st.caption(
            "In cyber incident classification, error impacts are highly asymmetric. "
            "Failing to recognize a critical incident creates extreme organizational exposure, whereas false alarms produce operational strain."
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            high_miss = error_risks["total_high_missed"]
            rate = error_risks["high_detection_rate"]
            st.markdown(
                f"""
                <div class="risk-box risk-critical">
                    <strong style="font-size:0.95rem;">🚨 Critical Threat Under-Classification</strong>
                    <div style="font-size:1.6rem;font-weight:800;margin:0.25rem 0;">{high_miss} Missed <span style="font-size:0.9rem;font-weight:400;">({44 - high_miss}/44 detected)</span></div>
                    <div style="font-size:0.78rem;">
                        <strong>High-Severity Recall: {rate:.1f}%</strong><br>
                        • Actual High predicted as Low: <b>{error_risks['high_as_low']}</b><br>
                        • Actual High predicted as Medium: <b>{error_risks['high_as_med']}</b><br>
                        <i>Risk: Critical breach goes unescalated, delaying response and containment.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:
            false_highs = error_risks["total_false_highs"]
            st.markdown(
                f"""
                <div class="risk-box risk-warning">
                    <strong style="font-size:0.95rem;">⚠️ False High Alarms (Over-Escalation)</strong>
                    <div style="font-size:1.6rem;font-weight:800;margin:0.25rem 0;">{false_highs} Over-predicted</div>
                    <div style="font-size:0.78rem;">
                        • Actual Low predicted as High: <b>{error_risks['low_as_high']}</b><br>
                        • Actual Medium predicted as High: <b>{error_risks['med_as_high']}</b><br>
                        <i>Risk: CSIRT alert fatigue, diversion of emergency incident handling resources from true threats.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r3:
            cross_lm = error_risks["med_as_low"] + error_risks["low_as_med"]
            st.markdown(
                f"""
                <div class="risk-box risk-info">
                    <strong style="font-size:0.95rem;">ℹ️ Moderate / Low Boundary Shifts</strong>
                    <div style="font-size:1.6rem;font-weight:800;margin:0.25rem 0;">{cross_lm} Borderline Shifts</div>
                    <div style="font-size:0.78rem;">
                        • Actual Medium predicted as Low: <b>{error_risks['med_as_low']}</b><br>
                        • Actual Low predicted as Medium: <b>{error_risks['low_as_med']}</b><br>
                        <i>Operational impact: Minor routing delays; manageable through Tier-1/2 triage queues.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("📖 False Positives vs. False Negatives Reference Guide"):
            st.markdown("""
            | Severity Tier | False Positive (FP) Definition & Impact | False Negative (FN) Definition & Impact |
            | :--- | :--- | :--- |
            | **High / Critical** | **Non-High incident classified as High.** Triggers emergency alerts, page-outs, and executive escalation for low/moderate impacts. Leads to SOC fatigue. | **Critical incident classified as Low or Medium.** The most dangerous failure mode in cybersecurity. Allows active intrusions or severe data destruction to proceed uncontained. |
            | **Medium** | **Low or High incident classified as Medium.** Over-allocates resources to routine noise or under-allocates resources to critical intrusions. | **Medium incident classified as Low or High.** May cause minor triage routing issues or queue delays. |
            | **Low** | **Medium or High incident classified as Low.** Severe danger if High is misclassified as Low (undetected adversary). | **Low incident classified as Medium or High.** Over-investigation of benign or minor probes. |
            """)

        # Cross-model error comparison table
        st.subheader("Cross-Model Error Comparison (Held-Out Test Set)")
        st.caption("Side-by-side comparison of error distributions across all 6 tuned model architectures.")
        
        comp_records = []
        for m_name in available_models:
            m_preds = test_predictions[test_predictions["model"] == m_name]
            _, m_summary = compute_detailed_metrics(m_preds)
            m_risks = analyze_error_risks(m_preds)
            comp_records.append({
                "Model": m_name,
                "Accuracy": m_summary["Accuracy"],
                "Macro F1": m_summary["Macro F1"],
                "High Severity Recall": m_risks["high_detection_rate"] / 100.0,
                "High Missed (FN)": m_risks["total_high_missed"],
                "High as Low": m_risks["high_as_low"],
                "High as Medium": m_risks["high_as_med"],
                "False Highs (FP)": m_risks["total_false_highs"]
            })
        
        df_comp = pd.DataFrame(comp_records).sort_values("Macro F1", ascending=False)
        st.dataframe(
            df_comp.style.format({
                "Accuracy": "{:.2%}",
                "Macro F1": "{:.4f}",
                "High Severity Recall": "{:.1%}",
                "High Missed (FN)": "{:d}",
                "High as Low": "{:d}",
                "High as Medium": "{:d}",
                "False Highs (FP)": "{:d}"
            }),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 2: HELD-OUT TEST EVALUATION
# ==========================================
with tab_heldout:
    st.subheader("Tuned Models — Held-Out Test Set Performance")
    test_metric = st.selectbox(
        "Held-out test metric",
        options=[m for m in available_metrics if m in test.columns],
        index=([m for m in available_metrics if m in test.columns].index("macro_f1")
               if "macro_f1" in test.columns else 0),
        format_func=lambda x: metric_labels.get(x, x),
        key="test_metric_select",
    )
    test_plot = test[["model", test_metric]].copy().sort_values(test_metric, ascending=True)
    fig_test = px.bar(test_plot, x=test_metric, y="model", orientation="h", text=test_metric)
    fig_test.update_traces(marker_color="#4F8CFF", texttemplate="%{text:.3f}", textposition="outside")
    fig_test.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=45, t=20, b=10), height=350,
        xaxis=dict(title=metric_labels.get(test_metric, test_metric), gridcolor="#20324B"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_test, use_container_width=True)

    st.dataframe(
        test.sort_values("macro_f1", ascending=False).style.format({
            col: "{:.4f}" for col in test.select_dtypes(include="number").columns
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download held-out test results CSV",
        test.to_csv(index=False).encode("utf-8-sig"),
        "tuned_test_results.csv",
        "text/csv"
    )

# ==========================================
# TAB 3: 12-MODEL CROSS-VALIDATION
# ==========================================
with tab_cv:
    st.subheader("Cross-Validation Comparison — 12 Candidate Models")
    cv_metric = st.selectbox(
        "Metric to compare across CV folds",
        options=available_metrics,
        index=available_metrics.index("macro_f1") if "macro_f1" in available_metrics else 0,
        format_func=lambda x: metric_labels.get(x, x),
        key="cv_metric_select",
    )
    cv_plot = cv[["model", cv_metric]].copy().sort_values(cv_metric, ascending=True)
    fig_cv = px.bar(cv_plot, x=cv_metric, y="model", orientation="h", text=cv_metric)
    fig_cv.update_traces(marker_color="#35D0E8", texttemplate="%{text:.3f}", textposition="outside")
    fig_cv.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=45, t=20, b=10), height=470,
        xaxis=dict(title=metric_labels.get(cv_metric, cv_metric), gridcolor="#20324B"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    with st.expander("View Full 12-Model Cross-Validation Metrics Table"):
        st.dataframe(cv.sort_values("macro_f1", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "Download CV results CSV",
            cv.to_csv(index=False).encode("utf-8-sig"),
            "cv_12_model_results.csv",
            "text/csv"
        )

# ==========================================
# TAB 4: TUNED HYPERPARAMETERS
# ==========================================
with tab_params:
    st.subheader("Best Tuned Hyperparameter Configurations")
    st.caption("Settings read directly from the DA2 hyperparameter tuning output; no models are retrained in Python.")
    show_cols = [c for c in ["model", "macro_f1", "mtry", "trees", "min_n", "tree_depth",
                             "learn_rate", "loss_reduction", "sample_size", "cost_complexity",
                             "hidden_units", "penalty", "epochs"] if c in params.columns]
    st.dataframe(params[show_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Download best hyperparameters CSV",
        params.to_csv(index=False).encode("utf-8-sig"),
        "best_hyperparameters.csv",
        "text/csv"
    )

st.info(
    "Evaluation Methodology Note: Cross-validation metrics and held-out test metrics derive from distinct validation stages. "
    "Held-out results represent unbiased generalization estimates on unobserved incidents."
)
