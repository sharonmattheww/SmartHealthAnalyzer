from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DEPARTMENTS = ["Engineering", "Commerce", "Arts"]
STUDENT_TYPES = ["Hostel", "Day Scholar"]
CAMPUS_ZONES = ["Main Block", "Library", "Auditorium", "Hostel", "Sports Complex"]


def clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def calculate_stress_index(
    heart_rate: float,
    systolic_bp: float,
    diastolic_bp: float,
    spo2: float,
    hydration: float,
    aqi: float,
) -> float:
    hr_component = min(abs(heart_rate - 75) / 35, 1.5)
    bp_component = min((abs(systolic_bp - 120) / 35 + abs(diastolic_bp - 80) / 20) / 2, 1.5)
    spo2_component = min(max((96 - spo2) / 5, 0), 1.5)
    hydration_component = min(max((55 - hydration) / 35, 0), 1.5)
    aqi_component = min(max((aqi - 90) / 90, 0), 1.5)
    stress_score = (0.30 * hr_component + 0.20 * bp_component + 0.20 * spo2_component + 0.15 * hydration_component + 0.15 * aqi_component) * 100
    return clamp(stress_score, 0, 100)


def calculate_health_score(
    heart_rate: float,
    systolic_bp: float,
    diastolic_bp: float,
    temperature: float,
    spo2: float,
    stress_index: float,
) -> float:
    hr_abnormality = min(abs(heart_rate - 75) / 40, 1)
    bp_abnormality = min((abs(systolic_bp - 120) / 40 + abs(diastolic_bp - 80) / 25) / 2, 1)
    temp_abnormality = min(abs(temperature - 98.4) / 2.5, 1)
    spo2_abnormality = min(max((97 - spo2) / 6, 0), 1)
    stress_component = min(stress_index / 100, 1)

    risk_value = (
        0.22 * hr_abnormality
        + 0.22 * bp_abnormality
        + 0.18 * temp_abnormality
        + 0.20 * spo2_abnormality
        + 0.18 * stress_component
    )

    score = 100 - (risk_value * 100)
    return clamp(score, 0, 100)


def get_risk_level(health_score: float) -> str:
    if health_score >= 70:
        return "Healthy"
    if health_score >= 45:
        return "Warning"
    return "Critical"


def generate_health_record(previous_row: Optional[Dict] = None) -> Dict:
    department = np.random.choice(DEPARTMENTS, p=[0.45, 0.30, 0.25])
    student_type = np.random.choice(STUDENT_TYPES, p=[0.55, 0.45])
    campus_zone = np.random.choice(CAMPUS_ZONES)

    weather_temp_c = np.random.normal(29, 3)
    humidity = np.random.normal(63, 10)
    zone_aqi_shift = {
        "Main Block": 5,
        "Library": -4,
        "Auditorium": 8,
        "Hostel": 12,
        "Sports Complex": 0,
    }[campus_zone]
    aqi = clamp(np.random.normal(88, 18) + zone_aqi_shift + (humidity - 60) * 0.2, 40, 210)

    dept_stress_shift = {"Engineering": 7, "Commerce": 3, "Arts": 0}[department]
    type_stress_shift = {"Hostel": 4, "Day Scholar": -2}[student_type]

    previous_hr = previous_row["heart_rate"] if previous_row else np.random.normal(78, 8)
    previous_spo2 = previous_row["spo2"] if previous_row else np.random.normal(97, 1.1)

    heart_rate = clamp(np.random.normal(previous_hr, 5) + dept_stress_shift * 0.15 + type_stress_shift * 0.2, 48, 155)
    systolic_bp = clamp(np.random.normal(118 + dept_stress_shift * 0.5, 11), 88, 185)
    diastolic_bp = clamp(np.random.normal(78 + dept_stress_shift * 0.4, 8), 55, 120)
    temperature_f = clamp(np.random.normal(98.4 + (weather_temp_c - 28) * 0.03, 0.55), 96.4, 103.2)
    spo2 = clamp(np.random.normal(previous_spo2, 0.9) - max((aqi - 120) / 120, 0.0), 85, 100)
    hydration = clamp(np.random.normal(66 - (weather_temp_c - 28) * 1.2 - dept_stress_shift * 0.2, 10), 20, 100)

    stress_index = calculate_stress_index(
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        spo2=spo2,
        hydration=hydration,
        aqi=aqi,
    )

    health_score = calculate_health_score(
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        temperature=temperature_f,
        spo2=spo2,
        stress_index=stress_index,
    )

    risk_level = get_risk_level(health_score)

    return {
        "timestamp": datetime.now(),
        "student_id": int(np.random.randint(1000, 1300)),
        "department": department,
        "student_type": student_type,
        "campus_zone": campus_zone,
        "heart_rate": round(heart_rate, 1),
        "systolic_bp": round(systolic_bp, 1),
        "diastolic_bp": round(diastolic_bp, 1),
        "temperature_f": round(temperature_f, 2),
        "spo2": round(spo2, 1),
        "hydration": round(hydration, 1),
        "aqi": round(aqi, 1),
        "weather_temp_c": round(weather_temp_c, 1),
        "humidity": round(clamp(humidity, 25, 95), 1),
        "stress_index": round(stress_index, 1),
        "health_score": round(health_score, 1),
        "risk_level": risk_level,
    }


def _to_float(value: object, fallback: float) -> float:
    if value is None:
        return float(fallback)
    try:
        parsed = float(value)
        if np.isnan(parsed):
            return float(fallback)
        return parsed
    except (TypeError, ValueError):
        return float(fallback)


def _to_int(value: object, fallback: int) -> int:
    if value is None:
        return int(fallback)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(fallback)


def normalize_ingested_record(record: Dict, previous_row: Optional[Dict] = None) -> Dict:
    fallback = previous_row or {}

    department = str(record.get("department", fallback.get("department", "Engineering")))
    if department not in DEPARTMENTS:
        department = "Engineering"

    student_type = str(record.get("student_type", fallback.get("student_type", "Day Scholar")))
    if student_type not in STUDENT_TYPES:
        student_type = "Day Scholar"

    campus_zone = str(record.get("campus_zone", fallback.get("campus_zone", "Main Block")))
    if campus_zone not in CAMPUS_ZONES:
        campus_zone = "Main Block"

    heart_rate = clamp(_to_float(record.get("heart_rate"), _to_float(fallback.get("heart_rate"), 78.0)), 48, 155)
    systolic_bp = clamp(_to_float(record.get("systolic_bp"), _to_float(fallback.get("systolic_bp"), 118.0)), 88, 185)
    diastolic_bp = clamp(_to_float(record.get("diastolic_bp"), _to_float(fallback.get("diastolic_bp"), 78.0)), 55, 120)
    temperature_f = clamp(_to_float(record.get("temperature_f"), _to_float(fallback.get("temperature_f"), 98.4)), 96.4, 103.2)
    spo2 = clamp(_to_float(record.get("spo2"), _to_float(fallback.get("spo2"), 97.0)), 85, 100)
    hydration = clamp(_to_float(record.get("hydration"), _to_float(fallback.get("hydration"), 66.0)), 20, 100)
    aqi = clamp(_to_float(record.get("aqi"), _to_float(fallback.get("aqi"), 88.0)), 40, 210)
    weather_temp_c = clamp(_to_float(record.get("weather_temp_c"), _to_float(fallback.get("weather_temp_c"), 29.0)), 10, 48)
    humidity = clamp(_to_float(record.get("humidity"), _to_float(fallback.get("humidity"), 63.0)), 25, 95)

    stress_index = calculate_stress_index(
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        spo2=spo2,
        hydration=hydration,
        aqi=aqi,
    )

    health_score = calculate_health_score(
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        temperature=temperature_f,
        spo2=spo2,
        stress_index=stress_index,
    )

    raw_timestamp = record.get("timestamp", datetime.now())
    timestamp = pd.to_datetime(raw_timestamp, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now()

    student_id = _to_int(record.get("student_id"), _to_int(fallback.get("student_id"), 1000))

    return {
        "timestamp": timestamp.to_pydatetime(),
        "student_id": student_id,
        "department": department,
        "student_type": student_type,
        "campus_zone": campus_zone,
        "heart_rate": round(heart_rate, 1),
        "systolic_bp": round(systolic_bp, 1),
        "diastolic_bp": round(diastolic_bp, 1),
        "temperature_f": round(temperature_f, 2),
        "spo2": round(spo2, 1),
        "hydration": round(hydration, 1),
        "aqi": round(aqi, 1),
        "weather_temp_c": round(weather_temp_c, 1),
        "humidity": round(humidity, 1),
        "stress_index": round(stress_index, 1),
        "health_score": round(health_score, 1),
        "risk_level": get_risk_level(health_score),
    }


def normalize_ingested_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    rows = []
    previous = None
    for row in raw_df.to_dict(orient="records"):
        normalized = normalize_ingested_record(row, previous_row=previous)
        rows.append(normalized)
        previous = normalized
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def build_student_baselines(df: pd.DataFrame, min_points: int = 5) -> pd.DataFrame:
    if df is None or df.empty or "student_id" not in df.columns:
        return pd.DataFrame()

    metrics = ["heart_rate", "spo2", "stress_index"]
    available_metrics = [col for col in metrics if col in df.columns]
    if not available_metrics:
        return pd.DataFrame()

    baseline_frames = []
    grouped = df.groupby("student_id")
    for metric in available_metrics:
        quantiles = grouped[metric].quantile([0.1, 0.5, 0.9]).unstack(level=1)
        quantiles.columns = [f"{metric}_p10", f"{metric}_p50", f"{metric}_p90"]
        baseline_frames.append(quantiles)

    baseline = pd.concat(baseline_frames, axis=1)
    baseline["record_count"] = grouped.size()
    baseline = baseline[baseline["record_count"] >= min_points]
    return baseline.reset_index()


def personalized_alert_summary(latest_row: pd.Series, student_baseline: Optional[pd.Series]) -> Tuple[str, list[str]]:
    if student_baseline is None:
        return "info", ["Not enough history to compute personalized baseline yet (minimum 5 records per student)."]

    checks = [
        ("heart_rate", "Heart rate"),
        ("spo2", "SpO₂"),
        ("stress_index", "Stress index"),
    ]

    findings = []
    severity = "info"
    for key, label in checks:
        low_key = f"{key}_p10"
        high_key = f"{key}_p90"
        median_key = f"{key}_p50"
        value = float(latest_row[key])
        low = float(student_baseline[low_key])
        high = float(student_baseline[high_key])
        median = float(student_baseline[median_key])

        if value < low:
            findings.append(f"{label} is below personal baseline ({value:.1f} vs expected {low:.1f}–{high:.1f}, median {median:.1f}).")
            severity = "warning" if severity == "info" else severity
        elif value > high:
            findings.append(f"{label} is above personal baseline ({value:.1f} vs expected {low:.1f}–{high:.1f}, median {median:.1f}).")
            severity = "error" if key in {"stress_index", "heart_rate"} else "warning"

    if not findings:
        findings.append("Current vitals are within the student’s personalized baseline range.")

    return severity, findings
