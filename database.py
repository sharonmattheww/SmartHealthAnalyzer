import os
from pathlib import Path

import pandas as pd
from supabase import Client, create_client


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_ANON_KEY")

if not URL or not KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")

supabase: Client = create_client(URL, KEY)

def save_health_record(record: dict):
    # This sends your dictionary directly to the cloud table
    # We convert 'timestamp' to string because JSON doesn't like datetime objects
    payload = record.copy()
    payload["timestamp"] = str(payload["timestamp"])
    response = supabase.table("student_health_logs").insert(payload).execute()
    return response.data

def fetch_latest_logs(limit: int = 50):
    # This pulls data back for your Plotly charts
    response = supabase.table("student_health_logs").select("*").order("created_at", desc=True).limit(limit).execute()
    return response.data

def fetch_historical_trends(student_id: int = None, department: str = None, limit: int = 1000):
    query = supabase.table("student_health_logs").select("*")
    
    # Apply filters if they are selected in the sidebar
    if student_id:
        query = query.eq("student_id", student_id)
    if department and department != "All":
        query = query.eq("department", department)
        
    response = query.order("timestamp", desc=False).limit(limit).execute()
    return pd.DataFrame(response.data)