import streamlit as st
import pandas as pd
from src.database import load_incidents, connection_status
from src.schema import detect_columns
from src.geo import compute_geo_summary, build_choropleth_figure

st.set_page_config(page_title="Incident Explorer & Global Heat Map", page_icon="🌐", layout="wide")
st.markdown("""
<style>
.stApp { background:#08111F; color:#E7EEF8; font-family:Inter,sans-serif; }
[data-testid="stSidebar"] { background:#0B1728; border-right:1px solid #20324B; }
.eyebrow { color:#35D0E8; font-family:monospace; letter-spacing:.15em; font-size:.72rem; }
.metric-box { background:linear-gradient(135deg,#12233A,#0E1A2B); border:1px solid #263A55; border-radius:12px; padding:0.9rem; }
.metric-box .val { font-size:1.6rem; font-weight:700; color:#E7EEF8; margin-top:0.2rem; }
.metric-box .lbl { font-size:0.75rem; text-transform:uppercase; color:#8FA3BD; letter-spacing:0.06em; }
.metric-box .sub { font-size:0.72rem; color:#35D0E8; margin-top:0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">INVESTIGATION WORKSPACE // GEOGRAPHIC THREAT INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("Incident Explorer & Threat Heat Map")
st.caption("Investigate cyber incident distributions across geography, time, incident types, and severity classes.")

ok, detail = connection_status()
if not ok:
    st.error("MySQL connection failed")
    st.code(detail)
    st.stop()

try:
    df = load_incidents()
except Exception as exc:
    st.error("Could not load the incident table")
    st.code(str(exc))
    st.stop()

if df.empty:
    st.warning("No records found.")
    st.stop()

cols = detect_columns(df)
year_col = cols["year"]
sev_col = cols["severity"]
type_col = cols["type"]
country_col = cols["country"]

filtered = df.copy()

with st.sidebar:
    st.markdown("## 🌐 CYBER INTEL")
    st.caption("GLOBAL INCIDENT MONITOR")
    st.page_link("app.py", label="Overview", icon=":material/dashboard:")
    st.page_link("pages/1_Incident_Explorer.py", label="Incident Explorer", icon=":material/search:")
    st.page_link("pages/2_Model_Performance.py", label="Model Performance Lab", icon=":material/monitoring:")
    st.page_link("pages/3_Feature_Insights.py", label="Feature Insights", icon=":material/insights:")
    st.page_link("pages/4_Severity_Prediction.py", label="Severity Prediction Demo", icon=":material/science:")
    st.divider()
    st.subheader("Filter Incidents")
    
    if year_col:
        years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        if not years.empty:
            yr = st.slider("Year range", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
            yn = pd.to_numeric(filtered[year_col], errors="coerce")
            filtered = filtered[yn.between(*yr)]
            
    if sev_col:
        values = sorted(filtered[sev_col].dropna().astype(str).unique())
        chosen_sev = st.multiselect("Severity", values, default=values)
        filtered = filtered[filtered[sev_col].astype(str).isin(chosen_sev)]
        
    if type_col:
        values = sorted(filtered[type_col].dropna().astype(str).unique())
        chosen_types = st.multiselect("Incident type", values, default=values)
        filtered = filtered[filtered[type_col].astype(str).isin(chosen_types)]
        
    if country_col:
        values = sorted(filtered[country_col].dropna().astype(str).unique())
        chosen_countries = st.multiselect("Country / location filter", values, default=[])
        if chosen_countries:
            filtered = filtered[filtered[country_col].astype(str).isin(chosen_countries)]
            
    query = st.text_input("Search raw text fields", placeholder="Search actors, targets, types…")
    if query.strip():
        text_cols = filtered.select_dtypes(include=["object", "string"]).columns.tolist()
        if text_cols:
            mask = filtered[text_cols].fillna("").astype(str).apply(lambda s: s.str.contains(query, case=False, regex=False)).any(axis=1)
            filtered = filtered[mask]

# Compute geographic summary on current filtered view
agg_geo, geo_stats, unmapped_breakdown = compute_geo_summary(filtered, country_col=country_col, sev_col=sev_col, type_col=type_col)

# Overview KPIs
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-box"><div class="lbl">Filtered Incidents</div><div class="val">{len(filtered):,}</div><div class="sub">Active query records</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-box"><div class="lbl">Mappable Attributions</div><div class="val">{geo_stats.get("mappable", 0):,}</div><div class="sub">{geo_stats.get("pct_mappable", 0.0):.1f}% with country origin</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-box"><div class="lbl">Attributed Origin States</div><div class="val">{geo_stats.get("distinct_countries", 0):,}</div><div class="sub">Distinct ISO-3 nations</div></div>', unsafe_allow_html=True)
with m4:
    unmap_count = geo_stats.get("unmappable", 0)
    st.markdown(f'<div class="metric-box"><div class="lbl">Unattributed / Regional</div><div class="val">{unmap_count:,}</div><div class="sub">{100.0 - geo_stats.get("pct_mappable", 0.0):.1f}% unknown / non-state</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

tab_map, tab_records = st.tabs(["🌍 Global Incident Heat Map", "📋 Incident Records Explorer"])

with tab_map:
    map_ctrl1, map_ctrl2 = st.columns([1.5, 2.5])
    with map_ctrl1:
        view_mode = st.selectbox(
            "Heat Map Layer View",
            options=["Total Incident Count", "High Severity Count", "Severity Proportion (% High)", "Dominant Severity"],
            index=0,
            help="Switch between total attributed incident volumes and severity-specific perspectives."
        )
    with map_ctrl2:
        st.caption(
            "Visualizing attributed initiator country distribution. Darker/brighter color intensities reflect higher threat activity. "
            "Hover over any country for detailed breakdowns of severity tiers and dominant attack vectors."
        )

    # Render Choropleth Map
    fig_map = build_choropleth_figure(agg_geo, view_mode=view_mode)
    st.plotly_chart(fig_map, use_container_width=True)

    c_left, c_right = st.columns([1.6, 1.0])
    with c_left:
        st.subheader("Top Attributed Threat Origin States")
        if not agg_geo.empty:
            display_cols = ["iso3", "country_name", "total_incidents", "low_count", "medium_count", "high_count", "pct_high", "dominant_severity"]
            st.dataframe(
                agg_geo[display_cols].rename(columns={
                    "iso3": "ISO-3",
                    "country_name": "Country",
                    "total_incidents": "Total Incidents",
                    "low_count": "Low",
                    "medium_count": "Medium",
                    "high_count": "High / Critical",
                    "pct_high": "% High",
                    "dominant_severity": "Dominant Severity"
                }).style.format({"% High": "{:.1f}%", "Total Incidents": "{:,}", "Low": "{:,}", "Medium": "{:,}", "High / Critical": "{:,}"}),
                use_container_width=True,
                hide_index=True,
                height=320
            )
        else:
            st.info("No country records match the current filter criteria.")

    with c_right:
        st.subheader("Attribution & Governance Limits")
        st.info(
            "**Data Governance & Geographic Integrity Notice**:\n\n"
            "- **Initiator Geography**: Tracks recorded country origin of *attributed attackers*. Out of the total dataset, "
            f"**{geo_stats.get('mappable', 0):,} records ({geo_stats.get('pct_mappable', 0.0):.1f}%)** have confirmed national attribution.\n\n"
            "- **Victim Geography**: Target locations in `receiver_regions` represent political and defense blocs "
            "(e.g., `['NATO', 'NORTHAM']`, `['EUROPE', 'EU', 'WESTEU']`, `['CSTO', 'SCO']`). They cannot be mapped to individual country ISO-3 codes without inventing false coordinates.\n\n"
            "- **Unattributed Incidents**: Unattributed or macro-regional incidents are never fabricated or guessed."
        )
        with st.expander(f"View Unattributed Breakdown ({unmap_count:,} records)"):
            if not unmapped_breakdown.empty:
                unmapped_df = unmapped_breakdown.reset_index()
                unmapped_df.columns = ["Attribution Value", "Incident Count"]
                st.dataframe(unmapped_df, use_container_width=True, hide_index=True)

with tab_records:
    st.subheader("Incident Records Table")
    st.caption("Direct table access with column sorting, text search, and CSV export.")
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=520)
    
    down_col1, down_col2 = st.columns([1.5, 2.5])
    with down_col1:
        st.download_button(
            "Download Filtered Incidents (CSV)",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            "filtered_cyber_incidents.csv",
            "text/csv",
            use_container_width=True
        )
    with down_col2:
        with st.expander("Detected Schema Mappings & Column Metadata"):
            st.json(cols)
            st.code("\n".join(map(str, df.columns)))
