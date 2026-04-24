import pandas as pd
from supabase import create_client, Client
from utils import normalize_ingested_dataframe
import datetime
import os
from dotenv import load_dotenv

# 1. Supabase Credentials (Replace with your actual keys)
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(URL, KEY)

def migrate():
    print("🚀 Starting data migration...")
    
    # 2. Load and Clean the CSV using your existing utils logic
    raw_df = pd.read_csv('sample_health_data.csv')
    normalized_df = normalize_ingested_dataframe(raw_df)
    
    # 3. Convert to a list of dictionaries for Supabase
    # We convert timestamps to strings so the JSON serializer understands them
    records = normalized_df.to_dict(orient='records')
    for record in records:
        if isinstance(record['timestamp'], (pd.Timestamp, datetime.datetime)):
            record['timestamp'] = record['timestamp'].isoformat()

    # 4. Upload to Supabase
    print(f"📦 Uploading {len(records)} records to Supabase...")
    try:
        response = supabase.table("student_health_logs").insert(records).execute()
        print("✅ Migration successful!")
    except Exception as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    migrate()