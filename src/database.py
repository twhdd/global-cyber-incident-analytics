"""Read-only MySQL connection helpers."""
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

def _config(key, default=None):
 value=os.getenv(key)
 if value: return value
 try: return st.secrets.get(key,default)
 except Exception: return default

def _table_name():
 table=_config("DB_TABLE","cyber_incident_data")
 if not table.replace("_","").isalnum(): raise ValueError("DB_TABLE must contain only letters, numbers and underscores")
 return table

@st.cache_resource
def get_engine():
 url=URL.create("mysql+mysqlconnector",username=_config("DB_USER","root"),password=_config("DB_PASSWORD",""),host=_config("DB_HOST","127.0.0.1"),port=int(_config("DB_PORT","3306")),database=_config("DB_NAME","cyber_incidents"),query={"charset":"utf8mb4"})
 return create_engine(url,pool_pre_ping=True,pool_recycle=1800)

def connection_status():
 try:
  with get_engine().connect() as conn: conn.execute(text("SELECT 1"))
  return True,"Connection successful."
 except Exception as exc: return False,str(exc)

@st.cache_data(ttl=300,show_spinner="Loading incident records from MySQL…")
def load_incidents():
 # Table identifier is validated; dashboard uses SELECT only.
 query=f"SELECT * FROM `{_table_name()}`"
 with get_engine().connect() as conn: return pd.read_sql_query(text(query),conn)
