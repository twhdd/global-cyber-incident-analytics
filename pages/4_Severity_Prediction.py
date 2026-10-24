import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.database import load_incidents, connection_status
from src.schema import detect_columns

load_dotenv()

st.set_page_config(
    page_title="Severity Prediction Demo",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background:#08111F; color:#E7EEF8; }
[data-testid="stSidebar"] { background:#0B1728; }
.eyebrow { color:#35D0E8; font-family:monospace; letter-spacing:.15em; font-size:.72rem; }
.prediction-panel { background:linear-gradient(135deg,#12233A,#0E1A2B); border:1px solid #263A55; border-radius:13px; padding:1.2rem; }
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

st.markdown('<div class="eyebrow">DA2 / INTERACTIVE INFERENCE</div>', unsafe_allow_html=True)
st.title("Severity Prediction Demo")
st.caption(
    "Runs an existing tuned R model through Rscript. It does not retrain the model."
)

st.warning(
    "Demonstration only. Choose a real MySQL record as the baseline, then edit selected predictors. "
    "Unchanged predictors are inherited from that record. The output is a model estimate, not a causal explanation or operational security verdict."
)

connected, detail = connection_status()
if not connected:
    st.error("MySQL connection is not ready.")
    st.code(detail)
    st.stop()

try:
    df = load_incidents()
except Exception as exc:
    st.error("Could not load the MySQL incident table.")
    st.code(str(exc))
    st.stop()

if df.empty:
    st.warning("No records are available.")
    st.stop()

model_file = Path(__file__).resolve().parents[1] / "data" / "r_outputs" / "DA2_Tuned_Model_Objects.rds"
bridge_file = Path(__file__).resolve().parents[1] / "prediction_bridge.R"

def find_rscript():
    configured = os.getenv("RSCRIPT_PATH", "").strip().strip('"')
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("Rscript")
    if found:
        return found
    # Common Windows installation paths. If R is installed elsewhere, set RSCRIPT_PATH in .env.
    candidate_patterns = [
        (Path("C:/Program Files/R"), "*/bin/Rscript.exe"),
        (Path("C:/Program Files/R"), "*/bin/x64/Rscript.exe"),
        (Path("C:/R"), "*/bin/Rscript.exe"),
    ]
    for base_dir, pattern in candidate_patterns:
        if base_dir.exists():
            matches = sorted(base_dir.glob(pattern))
            if matches:
                return str(matches[-1])
    return None

rscript = find_rscript()

if not model_file.exists():
    st.error("The saved tuned-model RDS file is missing.")
    st.code(str(model_file))
    st.stop()

if not bridge_file.exists():
    st.error("prediction_bridge.R is missing from the project root.")
    st.stop()

if not rscript:
    st.error("Rscript was not found. R/RStudio must be installed to run genuine model predictions.")
    st.markdown(
        "Add the full path to `Rscript.exe` in your project `.env`, for example:"
    )
    st.code('RSCRIPT_PATH=C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe')
    st.caption("Use the path that actually exists on your computer. The prediction page will not substitute a rule-based guess.")
    st.stop()

cols = detect_columns(df)
year_col = cols.get("year")
type_col = cols.get("type")
country_col = cols.get("country")

# Model artifacts were saved from the R tuning workflow.
model_options = {
    "Extra Trees": "extra_trees",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "XGBoost": "xgboost",
    "Decision Tree": "decision_tree",
    "MLP": "mlp",
}

st.subheader("1. Choose a baseline record")
st.caption("The baseline supplies the complete set of model predictors required by the saved R workflow.")

row_labels = {}
for idx, row in df.iterrows():
    year = row.get(year_col, "Unknown") if year_col else "Unknown"
    incident_type = row.get(type_col, "Unknown") if type_col else "Unknown"
    country = row.get(country_col, "Unknown") if country_col else "Unknown"
    severity = row.get("severity_class", "Unknown")
    row_labels[idx] = f"Row {idx} · {year} · {str(incident_type)[:55]} · Initiator: {country} · Recorded class: {severity}"

selected_idx = st.selectbox(
    "MySQL incident row",
    options=list(df.index),
    format_func=lambda idx: row_labels[idx],
)

baseline = df.loc[selected_idx].copy()

with st.expander("Baseline record details"):
    st.dataframe(
        baseline.to_frame(name="Value").reset_index().rename(columns={"index": "Field"}),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("2. Adjust selected predictors")
with st.form("severity_prediction_form"):
    left, right = st.columns(2)

    with left:
        selected_model_name = st.selectbox("Model", list(model_options.keys()), index=0)
        if year_col:
            year_default = pd.to_numeric(pd.Series([baseline[year_col]]), errors="coerce").iloc[0]
            year_default = int(year_default) if pd.notna(year_default) else 2020
            new_year = st.number_input(
                "Incident year",
                min_value=2000,
                max_value=2024,
                value=max(2000, min(2024, year_default)),
                step=1,
            )
        else:
            new_year = None

        if type_col:
            type_values = sorted(df[type_col].dropna().astype(str).unique().tolist())
            current_type = str(baseline[type_col]) if pd.notna(baseline[type_col]) else None
            type_index = type_values.index(current_type) if current_type in type_values else 0
            new_type = st.selectbox("Incident type", type_values, index=type_index)
        else:
            new_type = None

        if country_col:
            country_values = sorted(df[country_col].dropna().astype(str).unique().tolist())
            current_country = str(baseline[country_col]) if pd.notna(baseline[country_col]) else None
            country_index = country_values.index(current_country) if current_country in country_values else 0
            new_country = st.selectbox("Initiator country", country_values, index=country_index)
        else:
            new_country = None

    with right:
        st.markdown("**Binary indicators**")
        binary_features = [
            "has_disruption", "is_disruption", "is_data_theft", "is_hijacking",
            "is_ransomware", "is_doxing", "target_critical_infrastructure",
            "target_government", "target_corporate", "target_media",
            "target_education", "target_social_groups", "attribution_available",
            "initiator_attributed", "state_affiliated_initiator", "phishing_used",
            "exploit_public_facing", "data_exfiltration", "mitre_data_encryption",
        ]
        changed_binary = {}
        available_binary = [c for c in binary_features if c in df.columns]
        for feature in available_binary:
            raw_value = pd.to_numeric(pd.Series([baseline[feature]]), errors="coerce").iloc[0]
            default_value = bool(raw_value == 1) if pd.notna(raw_value) else False
            changed_binary[feature] = st.checkbox(
                feature.replace("_", " ").title(),
                value=default_value,
                key=f"pred_{feature}",
            )

    submitted = st.form_submit_button("Run severity prediction", type="primary", use_container_width=True)

if submitted:
    scenario = baseline.copy()

    if year_col and new_year is not None:
        scenario[year_col] = int(new_year)
    if type_col and new_type is not None:
        scenario[type_col] = new_type
    if country_col and new_country is not None:
        scenario[country_col] = new_country

    for feature, enabled in changed_binary.items():
        scenario[feature] = int(enabled)

    # The target must never be included in inference input.
    scenario = scenario.drop(labels=["severity_class"], errors="ignore")
    input_df = pd.DataFrame([scenario])

    explanation_df = None
    try:
        with tempfile.TemporaryDirectory(prefix="cyber_severity_") as temp_dir:
            temp = Path(temp_dir)
            input_csv = temp / "prediction_input.csv"
            output_csv = temp / "prediction_output.csv"
            explanation_csv = temp / "prediction_explanation.csv"
            input_df.to_csv(input_csv, index=False)

            command = [
                rscript,
                str(bridge_file),
                str(model_file),
                model_options[selected_model_name],
                str(input_csv),
                str(output_csv),
                str(explanation_csv),
            ]

            with st.spinner(f"Running saved {selected_model_name} model and computing explanations via R…"):
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(bridge_file.parent),
                    check=False,
                )

            if result.returncode != 0:
                st.error("The R prediction process failed.")
                if result.stderr:
                    st.code(result.stderr[-6000:])
                if result.stdout:
                    with st.expander("R console output"):
                        st.code(result.stdout[-4000:])
                st.info(
                    "The saved model may require R packages from the original DA2 environment. "
                    "Run the DA2 tuning script's library setup in RStudio if a package is missing."
                )
                st.stop()

            if not output_csv.exists():
                st.error("R finished without producing a prediction file.")
                st.stop()

            prediction = pd.read_csv(output_csv)
            if explanation_csv.exists():
                try:
                    explanation_df = pd.read_csv(explanation_csv)
                except Exception:
                    explanation_df = None

        predicted_class = str(prediction.loc[0, "predicted_class"])
        probability_cols = [c for c in prediction.columns if c.startswith(".pred_")]
        probability_data = []
        for column in probability_cols:
            class_name = column.replace(".pred_", "")
            probability_data.append({
                "Severity": class_name,
                "Probability": float(prediction.loc[0, column]),
            })
        prob_df = pd.DataFrame(probability_data).sort_values("Probability", ascending=False)

        st.divider()
        st.subheader("3. Prediction result")
        color = {"High": "#FF647C", "Medium": "#F6B94A", "Low": "#53D6A2"}.get(predicted_class, "#35D0E8")
        st.markdown(
            f"""
            <div class="prediction-panel">
                <div class="eyebrow">MODEL OUTPUT / {selected_model_name.upper()}</div>
                <h2 style="margin:.4rem 0;color:{color};">Predicted severity: {predicted_class}</h2>
                <p style="color:#8FA3BD;margin:0;">Generated from the saved tuned R model and its fitted preprocessing workflow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_col, table_col = st.columns([1.2, 1])
        with chart_col:
            fig = px.bar(
                prob_df.sort_values("Probability"),
                x="Probability",
                y="Severity",
                orientation="h",
                text="Probability",
                range_x=[0, 1],
                color="Severity",
                color_discrete_map={"Low": "#53D6A2", "Medium": "#F6B94A", "High": "#FF647C"},
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=40, t=20, b=10),
                xaxis=dict(title="Predicted probability", tickformat=".0%", gridcolor="#20324B"),
                yaxis=dict(title=""),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with table_col:
            st.markdown("**Class probabilities**")
            st.dataframe(
                prob_df.assign(Probability=prob_df["Probability"].map(lambda x: f"{x:.2%}")),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download prediction result",
                prediction.to_csv(index=False).encode("utf-8-sig"),
                file_name="severity_prediction.csv",
                mime="text/csv",
            )

        st.caption(
            "Probabilities are model estimates, not calibrated guarantees. "
            "Predictions are generated directly by the fitted R model pipeline."
        )

        # ========================================================
        # SECTION 4: PREDICTION EXPLANATION & INFLUENTIAL FEATURES
        # ========================================================
        st.divider()
        st.subheader("4. Prediction Explanation & Influential Feature Breakdown")
        
        from src.explanation_view import clean_feature_name, build_local_explanation_figure

        global_imp_file = Path(__file__).resolve().parents[1] / "data" / "r_outputs" / "all_model_feature_importance.csv"
        global_imp = pd.read_csv(global_imp_file) if global_imp_file.exists() else None

        method_desc = "Local Model Explanation"
        if explanation_df is not None and not explanation_df.empty and "method" in explanation_df.columns:
            method_desc = str(explanation_df["method"].iloc[0])

        st.markdown(
            f"""
            <div style="background:#0F1F35;border:1px solid #20324B;border-radius:10px;padding:0.75rem 1rem;margin-bottom:1rem;">
                <span class="eyebrow" style="color:#35D0E8;">EXPLANATION ENGINE:</span>
                <strong style="color:#E7EEF8;margin-left:0.5rem;">{method_desc}</strong>
                <div style="font-size:0.78rem;color:#8FA3BD;margin-top:0.25rem;">
                    Local explanation for this specific instance using fitted <code>{selected_model_name}</code> decision boundaries.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab_local_exp, tab_global_comp = st.tabs(["🔍 Local Prediction Drivers", "🌐 Global Model Feature Importance"])

        with tab_local_exp:
            if explanation_df is not None and not explanation_df.empty:
                valid_exp = explanation_df[explanation_df["feature"] != "Diagnostic Note"]
                if not valid_exp.empty:
                    exp_chart_col, exp_table_col = st.columns([1.3, 1.2])
                    with exp_chart_col:
                        fig_local = build_local_explanation_figure(valid_exp, predicted_class, method_desc)
                        st.plotly_chart(fig_local, use_container_width=True)

                    with exp_table_col:
                        st.markdown(f"**Influential Feature Attributes for {predicted_class} Severity**")
                        display_exp = valid_exp.copy()
                        display_exp["Feature"] = display_exp["feature"].apply(clean_feature_name)
                        display_exp["Direction"] = display_exp["direction"]
                        display_exp["Impact"] = display_exp["contribution"]
                        
                        cols_to_show = ["Feature", "Impact", "Direction"]
                        st.dataframe(
                            display_exp[cols_to_show].style.format({"Impact": "{:+.4f}"}),
                            use_container_width=True,
                            hide_index=True,
                            height=330
                        )
                else:
                    st.info("Local explanation notes: " + str(explanation_df["direction"].iloc[0]))
            else:
                st.warning(
                    f"Local explanation calculation could not be computed for {selected_model_name}. "
                    "Check R console output for package details."
                )

        with tab_global_comp:
            st.markdown(f"**Global Feature Importance Prior ({selected_model_name})**")
            st.caption(
                "Global importance describes the model's overall average reliance on features across the entire 3,412-incident training dataset. "
                "Notice how local contributions for an individual incident can diverge from global rankings depending on specific active flags."
            )
            if global_imp is not None:
                # Filter to selected model
                model_match = global_imp[global_imp["model"].astype(str).str.lower() == selected_model_name.lower()]
                if model_match.empty:
                    # Try partial match or fallback to first available
                    model_match = global_imp.head(15)
                
                imp_col = "normalized_importance" if "normalized_importance" in model_match.columns else "importance"
                top_global = model_match.sort_values(imp_col, ascending=False).head(15).copy()
                top_global["Feature"] = top_global["feature"].apply(clean_feature_name)

                fig_global = px.bar(
                    top_global.sort_values(imp_col, ascending=True),
                    x=imp_col,
                    y="Feature",
                    orientation="h",
                    template="plotly_dark"
                )
                fig_global.update_traces(marker_color="#4F8CFF")
                fig_global.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=20, t=15, b=20),
                    height=350,
                    xaxis=dict(title=imp_col.replace("_", " ").title(), gridcolor="#20324B"),
                    yaxis=dict(title="")
                )
                st.plotly_chart(fig_global, use_container_width=True)
            else:
                st.info("Global feature importance artifact not found in data/r_outputs/.")

        # Methodological Notes & Governance
        st.markdown(
            """
            <div style="background:#081424;border:1px solid #1E344F;border-radius:10px;padding:0.9rem;margin-top:1rem;">
                <div style="color:#35D0E8;font-size:0.8rem;font-weight:600;margin-bottom:0.3rem;">🛡️ TARGET LEAKAGE SHIELD: VERIFIED</div>
                <div style="color:#8FA3BD;font-size:0.77rem;line-height:1.45;">
                    The ground truth target <code>severity_class</code> was strictly excluded from the inference matrix prior to running preprocessing and inference.
                    All predictors were supplied via baseline imputation and user scenario overrides.
                </div>
                <div style="color:#F6B94A;font-size:0.8rem;font-weight:600;margin-top:0.7rem;margin-bottom:0.3rem;">⚠️ SCIENTIFIC & OPERATIONAL INTERPRETATION CAVEAT</div>
                <div style="color:#8FA3BD;font-size:0.77rem;line-height:1.45;">
                    Feature contributions represent statistical associations and decision surface gradient directions within the fitted statistical model, 
                    <strong>NOT proof of real-world causation</strong>. Toggling an attribute in this simulator demonstrates what the mathematical model expects, 
                    not what will physically transpire in an active adversarial cyber campaign.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as exc:
        st.error("Could not complete the severity prediction.")
        st.code(str(exc))


