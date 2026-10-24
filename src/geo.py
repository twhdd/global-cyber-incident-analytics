"""Geographic parsing and mapping utilities for Global Cyber Incident Dashboard."""
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CYAN = "#35D0E8"
BLUE = "#4F8CFF"
MUTED = "#8FA3BD"
NAVY_BG = "#08111F"
NAVY_CARD = "#0E1A2B"
NAVY_BORDER = "#20324B"

# Comprehensive ISO-3 and standard country map for all distinct entities in initiator_country
ISO3_MAP = {
    'Russia': ('RUS', 'Russia'),
    'China': ('CHN', 'China'),
    "Korea, Democratic People's Republic of": ('PRK', 'North Korea'),
    'Iran, Islamic Republic of': ('IRN', 'Iran'),
    'Ukraine': ('UKR', 'Ukraine'),
    'United States': ('USA', 'United States'),
    'Turkey': ('TUR', 'Turkey'),
    'Pakistan': ('PAK', 'Pakistan'),
    'India': ('IND', 'India'),
    'Syria': ('SYR', 'Syria'),
    'Bangladesh': ('BGD', 'Bangladesh'),
    'Palestine': ('PSE', 'Palestine'),
    'United Kingdom': ('GBR', 'United Kingdom'),
    'Israel': ('ISR', 'Israel'),
    'Saudi Arabia': ('SAU', 'Saudi Arabia'),
    'Vietnam': ('VNM', 'Vietnam'),
    'Philippines': ('PHL', 'Philippines'),
    'Morocco': ('MAR', 'Morocco'),
    'Armenia': ('ARM', 'Armenia'),
    'Brazil': ('BRA', 'Brazil'),
    'Belarus': ('BLR', 'Belarus'),
    'Korea, Republic of': ('KOR', 'South Korea'),
    'Mexico': ('MEX', 'Mexico'),
    'Azerbaijan': ('AZE', 'Azerbaijan'),
    'Germany': ('DEU', 'Germany'),
    'Peru': ('PER', 'Peru'),
    'Malaysia': ('MYS', 'Malaysia'),
    'Kazakhstan': ('KAZ', 'Kazakhstan'),
    'Yemen': ('YEM', 'Yemen'),
    'Italy': ('ITA', 'Italy'),
    'Chile': ('CHL', 'Chile'),
    'Netherlands': ('NLD', 'Netherlands'),
    'Egypt': ('EGY', 'Egypt'),
    'Spain': ('ESP', 'Spain'),
    'Sudan': ('SDN', 'Sudan'),
    'Serbia': ('SRB', 'Serbia'),
    'Algeria': ('DZA', 'Algeria'),
    'Colombia': ('COL', 'Colombia'),
    'Lebanon': ('LBN', 'Lebanon'),
    'Indonesia': ('IDN', 'Indonesia'),
    'United Arab Emirates': ('ARE', 'United Arab Emirates'),
    'Myanmar': ('MMR', 'Myanmar'),
    'Venezuela': ('VEN', 'Venezuela'),
    'Thailand': ('THA', 'Thailand'),
    'Austria': ('AUT', 'Austria'),
    'Tunisia': ('TUN', 'Tunisia'),
    'Taiwan': ('TWN', 'Taiwan'),
    'Afghanistan': ('AFG', 'Afghanistan'),
    'Cambodia': ('KHM', 'Cambodia'),
    'Argentina': ('ARG', 'Argentina'),
    'France': ('FRA', 'France'),
    'Libya': ('LBY', 'Libya'),
    'Greece': ('GRC', 'Greece'),
    'Portugal': ('PRT', 'Portugal'),
    'Qatar': ('QAT', 'Qatar'),
    'Czech Republic': ('CZE', 'Czech Republic'),
    'Nigeria': ('NGA', 'Nigeria'),
    'Bahrain': ('BHR', 'Bahrain'),
    'Jordan': ('JOR', 'Jordan'),
    'Poland': ('POL', 'Poland'),
    'Togo': ('TGO', 'Togo'),
    'El Salvador': ('SLV', 'El Salvador'),
    'Albania': ('ALB', 'Albania'),
    'Canada': ('CAN', 'Canada'),
    'Dominican Republic': ('DOM', 'Dominican Republic'),
    'Australia': ('AUS', 'Australia'),
    'Switzerland': ('CHE', 'Switzerland'),
    'Cuba': ('CUB', 'Cuba'),
    'Ethiopia': ('ETH', 'Ethiopia'),
    'New Zealand': ('NZL', 'New Zealand'),
    'Paraguay': ('PRY', 'Paraguay'),
    'Uzbekistan': ('UZB', 'Uzbekistan'),
    'Romania': ('ROU', 'Romania'),
    'Kuwait': ('KWT', 'Kuwait'),
    'Guatemala': ('GTM', 'Guatemala'),
    'Slovakia': ('SVK', 'Slovakia'),
    'Latvia': ('LVA', 'Latvia'),
    'Japan': ('JPN', 'Japan'),
    'Panama': ('PAN', 'Panama'),
    'Iraq': ('IRQ', 'Iraq'),
    'Georgia': ('GEO', 'Georgia'),
    'Sweden': ('SWE', 'Sweden'),
    'Kyrgyzstan': ('KGZ', 'Kyrgyzstan'),
    'Ireland': ('IRL', 'Ireland'),
    'Iceland': ('ISL', 'Iceland'),
    'Finland': ('FIN', 'Finland'),
}

KNOWN_MULTI_NAMES = [
    "Korea, Democratic People's Republic of",
    "Iran, Islamic Republic of",
    "Korea, Republic of",
]

UNMAPPABLE_TERMS = {
    "not attributed", "unknown", "not available", "isis",
    "central america (region)", "south asia (region)", "middle east (region)",
    "asia (region)", "eastern europe", "southeast asia (region)",
    "south america", "europe (region)", "gulf countries (region)",
    "mena region (region)", "north america"
}

def parse_initiator_countries(raw_val):
    """
    Extracts validated (ISO-3, display_name) pairs from an initiator_country string.
    Safely ignores multi-nation macro regions and unattributed records.
    """
    if not raw_val or pd.isna(raw_val):
        return []
    s = str(raw_val).strip()
    if s.lower() in UNMAPPABLE_TERMS:
        return []
    
    # Temporarily substitute known comma-containing country names
    for idx, name in enumerate(KNOWN_MULTI_NAMES):
        s = s.replace(name, f"__MNAME_{idx}__")
    
    # Split across semicolons and commas
    tokens = re.split(r"[;,]", s)
    results = []
    seen_iso = set()
    
    for token in tokens:
        token = token.strip()
        for idx, name in enumerate(KNOWN_MULTI_NAMES):
            token = token.replace(f"__MNAME_{idx}__", name)
        
        # Check against ISO3_MAP
        if token in ISO3_MAP:
            iso, display_name = ISO3_MAP[token]
            if iso not in seen_iso:
                seen_iso.add(iso)
                results.append((iso, display_name))
    return results

def compute_geo_summary(df, country_col="initiator_country", sev_col="severity_class", type_col="incident_type"):
    """
    Aggregates filtered incidents by country ISO-3, computing total incidents,
    severity distributions, and unmappable statistics.
    """
    if country_col not in df.columns:
        return pd.DataFrame(), {"mappable": 0, "unmappable": len(df), "pct_mappable": 0.0}, pd.Series(dtype=int)
    
    parsed_col = df[country_col].apply(parse_initiator_countries)
    mappable_mask = parsed_col.apply(lambda l: len(l) > 0)
    mappable_count = int(mappable_mask.sum())
    unmappable_count = int((~mappable_mask).sum())
    pct_mappable = (mappable_count / len(df) * 100.0) if len(df) > 0 else 0.0

    unmappable_breakdown = df.loc[~mappable_mask, country_col].fillna("Missing / Null").value_counts()
    
    rows = []
    for idx, (is_map, country_pairs) in enumerate(zip(mappable_mask, parsed_col)):
        if not is_map:
            continue
        row_sev = str(df.iloc[idx].get(sev_col, "Unknown")) if sev_col and sev_col in df.columns else "Unknown"
        row_type = str(df.iloc[idx].get(type_col, "Unknown")) if type_col and type_col in df.columns else "Unknown"
        
        for iso, display_name in country_pairs:
            rows.append({
                "iso3": iso,
                "country_name": display_name,
                "severity": row_sev,
                "incident_type": row_type
            })
    
    if not rows:
        return pd.DataFrame(), {
            "mappable": mappable_count,
            "unmappable": unmappable_count,
            "pct_mappable": pct_mappable
        }, unmappable_breakdown
    
    expanded = pd.DataFrame(rows)
    
    # Aggregate by country
    summary_list = []
    for (iso, name), group in expanded.groupby(["iso3", "country_name"]):
        tot = len(group)
        low_c = int((group["severity"].str.lower() == "low").sum())
        med_c = int((group["severity"].str.lower() == "medium").sum())
        high_c = int(group["severity"].str.lower().isin(["high", "critical"]).sum())
        
        pct_high = (high_c / tot * 100.0) if tot > 0 else 0.0
        pct_med = (med_c / tot * 100.0) if tot > 0 else 0.0
        pct_low = (low_c / tot * 100.0) if tot > 0 else 0.0
        
        # Dominant severity
        counts_dict = {"Low": low_c, "Medium": med_c, "High": high_c}
        dominant = max(counts_dict, key=counts_dict.get) if tot > 0 else "Low"
        
        top_types = group["incident_type"].value_counts().head(2).index.tolist()
        top_types_str = ", ".join(top_types) if top_types else "None recorded"
        
        summary_list.append({
            "iso3": iso,
            "country_name": name,
            "total_incidents": tot,
            "low_count": low_c,
            "medium_count": med_c,
            "high_count": high_c,
            "pct_low": pct_low,
            "pct_medium": pct_med,
            "pct_high": pct_high,
            "dominant_severity": dominant,
            "top_types": top_types_str,
            "hover_text": (
                f"<b>{name}</b> ({iso})<br>"
                f"Total Attributed Incidents: <b>{tot:,}</b><br>"
                f"• Low: {low_c} ({pct_low:.1f}%)<br>"
                f"• Medium: {med_c} ({pct_med:.1f}%)<br>"
                f"• High / Critical: {high_c} ({pct_high:.1f}%)<br>"
                f"Dominant Severity: <b>{dominant}</b><br>"
                f"Top Attack Types: <i>{top_types_str}</i>"
            )
        })
    
    agg_df = pd.DataFrame(summary_list).sort_values("total_incidents", ascending=False)
    stats = {
        "mappable": mappable_count,
        "unmappable": unmappable_count,
        "pct_mappable": pct_mappable,
        "distinct_countries": len(agg_df),
        "total_attributions": int(agg_df["total_incidents"].sum()) if not agg_df.empty else 0
    }
    return agg_df, stats, unmappable_breakdown

def build_choropleth_figure(agg_df, view_mode="Total Incident Count"):
    """
    Renders an interactive Plotly choropleth figure styled for the dark navy dashboard.
    """
    if agg_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No mappable geographic records in current filter", showarrow=False, font=dict(color=MUTED, size=15))]
        )
        return fig

    if view_mode == "High Severity Count":
        val_col = "high_count"
        title = "High Severity Incident Count by Initiator Country"
        color_scale = ["#1A1020", "#4A1525", "#8A1C30", "#C72C41", "#FF647C"]
        c_label = "High Incidents"
    elif view_mode == "Severity Proportion (% High)":
        val_col = "pct_high"
        title = "Percentage of High-Severity Incidents by Country"
        color_scale = ["#0E2A3A", "#1C4E68", "#E69C24", "#FF647C"]
        c_label = "% High Severity"
    elif view_mode == "Dominant Severity":
        # Categorical choropleth
        color_map = {"Low": "#53D6A2", "Medium": "#F6B94A", "High": "#FF647C"}
        fig = px.choropleth(
            agg_df,
            locations="iso3",
            color="dominant_severity",
            hover_name="country_name",
            custom_data=["total_incidents", "low_count", "medium_count", "high_count", "top_types"],
            color_discrete_map=color_map,
            category_orders={"dominant_severity": ["Low", "Medium", "High"]},
            template="plotly_dark",
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Dominant Class: <b>%{color}</b><br>"
                "Total Incidents: %{customdata[0]:,}<br>"
                "Low: %{customdata[1]} | Med: %{customdata[2]} | High: %{customdata[3]}<br>"
                "Top Types: %{customdata[4]}<extra></extra>"
            )
        )
        _style_geo_layout(fig, "Dominant Incident Severity by Country")
        return fig
    else:  # Total Incident Count
        val_col = "total_incidents"
        title = "Total Attributed Incidents by Initiator Country"
        color_scale = ["#091D34", "#123C64", "#1F659C", "#2E91D3", CYAN]
        c_label = "Attributed Incidents"

    fig = go.Figure(
        data=go.Choropleth(
            locations=agg_df["iso3"],
            z=agg_df[val_col],
            text=agg_df["country_name"],
            customdata=agg_df[["total_incidents", "low_count", "medium_count", "high_count", "pct_high", "dominant_severity", "top_types"]],
            colorscale=color_scale,
            colorbar=dict(
                title=dict(text=c_label, font=dict(color=CYAN, size=11)),
                tickfont=dict(color="#E7EEF8", size=10),
                thickness=12,
                len=0.7,
                x=0.98,
                y=0.5,
                bgcolor="rgba(11,23,40,0.8)",
                bordercolor=NAVY_BORDER,
                borderwidth=1
            ),
            hovertemplate=(
                "<b>%{text}</b> (%{location})<br>"
                f"{c_label}: <b>%{{z:,}}</b><br>"
                "Total Attributed Incidents: %{customdata[0]:,}<br>"
                "• Low: %{customdata[1]}<br>"
                "• Medium: %{customdata[2]}<br>"
                "• High: %{customdata[3]} (%{customdata[4]:.1f}%)<br>"
                "Dominant Severity: <b>%{customdata[5]}</b><br>"
                "Primary Types: <i>%{customdata[6]}</i><extra></extra>"
            ),
            marker_line_color=NAVY_BORDER,
            marker_line_width=0.75,
        )
    )
    _style_geo_layout(fig, title)
    return fig

def _style_geo_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E7EEF8", size=14, family="Inter, sans-serif")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=35, b=0),
        height=480,
        geo=dict(
            bgcolor="rgba(8,17,31,0.0)",
            lakecolor="#08111F",
            landcolor="#0B192C",
            subunitcolor="#1F334E",
            countrycolor="#223755",
            showland=True,
            showocean=True,
            oceancolor="#070E1A",
            coastlinecolor="#243C5E",
            showlakes=True,
            projection_type="natural earth",
        )
    )
