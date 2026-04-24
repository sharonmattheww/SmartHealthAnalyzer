from fastapi import FastAPI
from database import fetch_latest_logs  # Use the function we created earlier
import pandas as pd

app = FastAPI()

@app.get("/analytics/{student_id}")
def get_student_analytics(student_id: int):
    # 1. Fetch data from Supabase
    raw_data = fetch_latest_logs(limit=20)
    df = pd.DataFrame(raw_data)
    
    # 2. Filter for the specific student
    student_df = df[df["student_id"] == student_id]
    
    if student_df.empty:
        return {"error": "No data found for this student"}

    # 3. Calculate a quick insight
    avg_health = student_df["health_score"].mean()
    
    return {
        "student_id": student_id,
        "average_health_score": round(avg_health, 2),
        "total_records": len(student_df)
    }