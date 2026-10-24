import streamlit as st
import pandas as pd
import plotly.express as px
from src.database import load_incidents, connection_status
from src.schema import detect_columns

st.set_page_config(page_title="Global Cyber Incident Intelligence", page_icon="🌐", layout="wide")
CYAN, BLUE, MUTED = "#35D0E8", "#4F8CFF", "#8FA3BD"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700;800&display=swap');
.stApp{{background:#08111F;color:#E7EEF8;font-family:Inter,sans-serif}}
[data-testid="stSidebar"]{{background:#0B1728;border-right:1px solid #20324B}}
[data-testid="stHeader"]{{background:rgba(8,17,31,.9)}}
.hero{{padding:1rem 0 1.2rem;border-bottom:1px solid #20324B;margin-bottom:1.2rem}}
.eyebrow{{color:{CYAN};font-family:'IBM Plex Mono',monospace;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase}}
.hero h1{{margin:.35rem 0;font-size:2.3rem;font-weight:800;letter-spacing:-.04em}}
.hero p,.note{{color:{MUTED}}}
.metric{{background:linear-gradient(135deg,#12233A,#0E1A2B);border:1px solid #263A55;border-radius:13px;padding:1rem;min-height:105px}}
.metric-label{{color:{MUTED};font-size:.76rem;text-transform:uppercase;letter-spacing:.08em}}
.metric-value{{font-size:1.85rem;font-weight:800;margin-top:.35rem}}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ◈ CYBER INTEL")
    st.caption("GLOBAL INCIDENT MONITOR")
    st.page_link("app.py", label="Overview", icon=":material/dashboard:")
    st.page_link("pages/1_Incident_Explorer.py", label="Incident Explorer", icon=":material/search:")
    st.page_link("pages/2_Model_Performance.py", label="Model Performance Lab", icon=":material/monitoring:")
    st.page_link("pages/3_Feature_Insights.py", label="Feature Insights", icon=":material/insights:")
    st.page_link("pages/4_Severity_Prediction.py", label="Severity Prediction Demo", icon=":material/science:")
    st.divider()
    st.caption("MYSQL · cyber_incidents")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

st.markdown('<div class="hero"><div class="eyebrow">PDS // Threat intelligence // 2000—2024</div><h1>Global Cyber Incident Intelligence</h1><p>Explore incident patterns, severity distribution, and geographic signals from the project dataset.</p></div>', unsafe_allow_html=True)
status, detail = connection_status()
if not status:
    st.error("MySQL connection is not ready.")
    st.code(detail)
    st.info("Copy .env.example to .env, enter your local MySQL credentials, then rerun the app.")
    st.stop()
try:
    df = load_incidents()
except Exception as exc:
    st.error("Connected to MySQL, but the incident table could not be loaded.")
    st.code(str(exc)); st.stop()
if df.empty:
    st.warning("The configured table returned zero rows."); st.stop()

cols = detect_columns(df)
year_col, sev_col, type_col, country_col = cols["year"], cols["severity"], cols["type"], cols["country"]
filtered = df.copy()
f1, f2, f3 = st.columns([1.15, 1.1, 1.5])
if year_col:
    years = pd.to_numeric(df[year_col], errors="coerce").dropna()
    if not years.empty:
        with f1: year_range = st.slider("Incident year", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
        yn = pd.to_numeric(filtered[year_col], errors="coerce")
        filtered = filtered[yn.between(*year_range)]
if sev_col:
    vals = sorted(df[sev_col].dropna().astype(str).unique())
    with f2: selected = st.multiselect("Severity", vals, default=vals)
    filtered = filtered[filtered[sev_col].astype(str).isin(selected)]
if type_col:
    vals = sorted(df[type_col].dropna().astype(str).unique())
    with f3: selected_types = st.multiselect("Incident type", vals, default=vals)
    filtered = filtered[filtered[type_col].astype(str).isin(selected_types)]

def metric(col, label, value, note=""):
    with col: st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)

n = len(filtered)
if sev_col:
    severity = filtered[sev_col].fillna("").astype(str).str.lower()
    high_count = int(severity.str.contains("high|critical", regex=True).sum())
else: high_count = None
k1,k2,k3,k4 = st.columns(4)
metric(k1,"Incidents in view",f"{n:,}","After filters")
metric(k2,"Incident types",f"{filtered[type_col].nunique():,}" if type_col else "—","Distinct values")
metric(k3,"Countries / locations",f"{filtered[country_col].nunique():,}" if country_col else "—","Distinct values")
metric(k4,"High / critical",f"{high_count:,}" if high_count is not None else "—",f"{high_count/n:.1%} of view" if n and high_count is not None else "Severity field not detected")

left,right = st.columns([1.4,1])
with left:
    st.subheader("Incident activity over time")
    if year_col:
        counts = pd.to_numeric(filtered[year_col], errors="coerce").dropna().astype(int).value_counts().sort_index()
        plot = px.area(pd.DataFrame({"Year":counts.index,"Incidents":counts.values}),x="Year",y="Incidents",template="plotly_dark")
        plot.update_traces(line_color=CYAN,fillcolor="rgba(53,208,232,.13)")
        plot.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=8,r=8,t=15,b=8),height=330,xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#20324B"))
        st.plotly_chart(plot,use_container_width=True)
    else: st.warning("Year column not detected. Add its name to src/schema.py.")
with right:
    st.subheader("Severity distribution")
    if sev_col:
        counts = filtered[sev_col].fillna("Unknown").astype(str).value_counts().rename_axis("Severity").reset_index(name="Incidents")
        plot = px.pie(counts,names="Severity",values="Incidents",hole=.67,template="plotly_dark",color_discrete_sequence=[CYAN,BLUE,"#F6B94A","#FF647C"])
        plot.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=15,b=5),height=330,legend=dict(orientation="h",y=-.12))
        plot.update_traces(textposition="inside",textinfo="percent")
        st.plotly_chart(plot,use_container_width=True)
    else: st.warning("Severity column not detected. Add its name to src/schema.py.")
left,right = st.columns(2)
with left:
    st.subheader("Most common incident types")
    if type_col:
        counts = filtered[type_col].fillna("Unknown").astype(str).value_counts().head(10).sort_values().reset_index()
        counts.columns=["Incident type","Incidents"]
        plot=px.bar(counts,x="Incidents",y="Incident type",orientation="h",template="plotly_dark")
        plot.update_traces(marker_color=BLUE); plot.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=10,t=10,b=5),height=350,xaxis=dict(gridcolor="#20324B"),yaxis_title="")
        st.plotly_chart(plot,use_container_width=True)
    else: st.info("Incident type column not detected.")
with right:
    st.subheader("Geographic distribution")
    if country_col:
        counts=filtered[country_col].fillna("Unknown").astype(str).value_counts().head(10).sort_values().reset_index(); counts.columns=["Country/location","Incidents"]
        plot=px.bar(counts,x="Incidents",y="Country/location",orientation="h",template="plotly_dark")
        plot.update_traces(marker_color=CYAN); plot.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=10,t=10,b=5),height=350,xaxis=dict(gridcolor="#20324B"),yaxis_title="")
        st.plotly_chart(plot,use_container_width=True)
    else: st.info("No country/location column detected; geography will not be inferred.")
st.divider(); st.subheader("Dataset health")
a,b,c=st.columns(3)
metric(a,"Columns loaded",f"{df.shape[1]:,}","Current MySQL schema")
metric(b,"Rows with missing values",f"{int(df.isna().any(axis=1).sum()):,}","At least one missing field")
metric(c,"Exact duplicate rows",f"{int(df.duplicated().sum()):,}","Across loaded columns")
with st.expander("Detected column mapping"): st.json(cols)
with st.expander("Preview raw records"): st.dataframe(filtered.head(100),use_container_width=True,hide_index=True)
st.caption("Read-only dashboard. R remains responsible for DA2 analysis and model training.")
