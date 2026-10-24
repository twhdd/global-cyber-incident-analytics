"""Column-name detection. Add actual schema names to these aliases if necessary."""
import re
ALIASES={
 "year":["incident_year","year","incident year","event_year","date_year"],
 "severity":["severity_class","severity","severity level","impact_severity"],
 "type":["incident_type","incident type","type","attack_type","event_type"],
 "country":["initiator_country","country","country_name","victim_country","location_country","nation"],
 "receiver_regions":["receiver_regions","receiver_region","target_regions","victim_regions"],
}
def normalize(name): return re.sub(r"[^a-z0-9]+","_",str(name).strip().lower()).strip("_")
def detect_columns(df):
 normalized={normalize(c):c for c in df.columns}; out={}
 for kind,aliases in ALIASES.items(): out[kind]=next((normalized[normalize(a)] for a in aliases if normalize(a) in normalized),None)
 return out

