from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Insights", page_icon="🔎", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "r_outputs"

st.markdown("""
<style>
.stApp { background:#08111F; color:#E7EEF8; }
[data-testid="stSidebar"] { background:#0B1728; }
.eyebrow { color:#35D0E8; font-family:monospace; letter-spacing:.15em; font-size:.72rem; }
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

st.markdown('<div class="eyebrow">DA2 / MODEL INTERPRETATION</div>', unsafe_allow_html=True)
st.title("Feature Insights")
st.caption("Explore existing R-generated feature-importance results. Importance indicates predictive association, not causation.")

def load_csv(filename):
    path = OUT / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None

consensus = load_csv("cross_model_feature_importance.csv")
all_importance = load_csv("all_model_feature_importance.csv")

if consensus is None or all_importance is None:
    st.error("Feature-importance CSVs are missing from data/r_outputs.")
    st.write("Expected files:")
    st.code("cross_model_feature_importance.csv\nall_model_feature_importance.csv")
    st.stop()

if not {"feature", "models_in_top20"}.issubset(consensus.columns):
    st.error("The consensus file must contain 'feature' and 'models_in_top20' columns.")
    st.stop()

consensus = consensus.copy()
consensus["models_in_top20"] = pd.to_numeric(consensus["models_in_top20"], errors="coerce")
consensus = consensus.dropna(subset=["models_in_top20"]).sort_values("models_in_top20", ascending=False)

top = consensus.iloc[0] if not consensus.empty else None
a, b, c = st.columns(3)
a.metric("Features in consensus file", f"{len(consensus):,}")
a.metric("Models in consensus", "6", "Six tuned/tree models in source output")
if top is not None:
    b.metric("Most consistent feature", str(top["feature"])[:32] + ("…" if len(str(top["feature"])) > 32 else ""))
    c.metric("Top-20 appearances", f"{int(top['models_in_top20'])}/6")
else:
    b.metric("Most consistent feature", "—")
    c.metric("Top-20 appearances", "—")

st.divider()
st.subheader("Cross-model feature consistency")
st.caption("Count of models in which each feature appeared in the top 20.")
top_n = st.slider("Number of features to display", min_value=5, max_value=min(47, max(5, len(consensus))), value=min(20, max(5, len(consensus))))
cons_view = consensus.head(top_n).sort_values("models_in_top20", ascending=True)
fig = px.bar(cons_view, x="models_in_top20", y="feature", orientation="h", text="models_in_top20")
fig.update_traces(marker_color="#35D0E8", texttemplate="%{text:.0f}/6", textposition="outside")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10,r=50,t=20,b=10), height=max(350, top_n * 24),
    xaxis=dict(title="Models with feature in top 20", range=[0, 6.8], dtick=1, gridcolor="#20324B"),
    yaxis=dict(title="", tickfont=dict(size=10)),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View consensus table"):
    st.dataframe(consensus, use_container_width=True, hide_index=True)
    st.download_button("Download consensus CSV", consensus.to_csv(index=False).encode("utf-8-sig"),
                       "cross_model_feature_importance.csv", "text/csv")

st.divider()
st.subheader("Feature importance by model")
required = {"feature", "model"}
if not required.issubset(all_importance.columns):
    st.error("The all-model importance file must contain 'feature' and 'model' columns.")
    st.stop()

importance_col = "normalized_importance" if "normalized_importance" in all_importance.columns else (
    "importance" if "importance" in all_importance.columns else None
)
if importance_col is None:
    st.error("No importance-value column was found.")
    st.stop()

all_importance[importance_col] = pd.to_numeric(all_importance[importance_col], errors="coerce")
all_importance = all_importance.dropna(subset=[importance_col, "feature", "model"])
models = sorted(all_importance["model"].astype(str).unique().tolist())
selected_model = st.selectbox("Select model", models)
top_k = st.slider("Features for selected model", min_value=5, max_value=30, value=15)
model_view = all_importance[all_importance["model"].astype(str) == selected_model]
model_view = model_view.sort_values(importance_col, ascending=False).head(top_k).sort_values(importance_col)

fig = px.bar(model_view, x=importance_col, y="feature", orientation="h")
fig.update_traces(marker_color="#4F8CFF")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10,r=15,t=20,b=10), height=max(350, top_k * 23),
    xaxis=dict(title=importance_col.replace("_", " ").title(), gridcolor="#20324B"),
    yaxis=dict(title="", tickfont=dict(size=10)),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View selected model's feature table"):
    st.dataframe(model_view.sort_values(importance_col, ascending=False), use_container_width=True, hide_index=True)

st.info(
    "Feature importance is model-dependent and describes how the fitted model uses predictors. "
    "It does not establish that a feature causes higher incident severity. "
    "Some incident-type features may be closely related to the target construction, so interpret them in the context of the DA2 leakage checks."
)
