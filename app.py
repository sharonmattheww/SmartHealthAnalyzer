from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    add_zscore_anomaly_flags,
    build_heatmap_matrix,
    moving_average_forecast,
    risk_projection_10min,
    rolling_regression_forecast,
)
from utils import DEPARTMENTS, STUDENT_TYPES, generate_health_record
from utils import (
    CAMPUS_ZONES,
    build_student_baselines,
    normalize_ingested_dataframe,
    normalize_ingested_record,
    personalized_alert_summary,
)


st.set_page_config(page_title="CHRIST Smart HealthBoard", page_icon="🩺", layout="wide")

st.title("🩺 CHRIST University Smart HealthBoard Monitoring Dashboard")
st.caption("Real-time campus health intelligence with risk detection, predictive analytics, and anomaly alerts.")

st.markdown(
    """
This dashboard simulates real-time student wellness data for CHRIST University and updates continuously.

- Streams vitals every 2–3 seconds with rolling data storage.
- Detects health risk levels (Healthy, Warning, Critical) using weighted scoring.
- Predicts near-future trends with moving average and rolling regression.
- Detects anomalies with Z-score and highlights campus intelligence patterns.
"""
)


def initialize_data(seed_size: int = 80) -> pd.DataFrame:
    rows = []
    previous = None
    for _ in range(seed_size):
        row = generate_health_record(previous_row=previous)
        rows.append(row)
        previous = row
    return pd.DataFrame(rows)


if "health_df" not in st.session_state:
    st.session_state.health_df = initialize_data(seed_size=80)
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = pd.DataFrame()
if "live_sensor_df" not in st.session_state:
    st.session_state.live_sensor_df = pd.DataFrame()


with st.sidebar:
    st.header("Dashboard Controls")
    data_source = st.selectbox("Data input mode", ["Simulation", "CSV Upload", "Live Sensor"]) 
    selected_department = st.selectbox("Select department", ["All"] + DEPARTMENTS)
    selected_student_type = st.selectbox("Select student type", ["All"] + STUDENT_TYPES)
    parameter = st.selectbox(
        "Select health parameter",
        ["heart_rate", "temperature_f", "spo2", "stress_index", "health_score", "aqi", "hydration"],
    )
    refresh_seconds = st.slider("Refresh interval (seconds)", min_value=2, max_value=3, value=2)
    rolling_window = st.slider("Rolling window size", min_value=50, max_value=100, value=80)
    auto_refresh = st.toggle("Auto refresh", value=True, disabled=data_source != "Simulation")

    if data_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload health CSV", type=["csv"])
        if uploaded_file is not None:
            incoming = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = normalize_ingested_dataframe(incoming)
            st.success(f"CSV loaded with {len(st.session_state.uploaded_df)} normalized records.")

    if data_source == "Live Sensor":
        st.markdown("### Ingest Live Sensor Reading")
        with st.form("live_sensor_form"):
            live_student_id = st.number_input("Student ID", min_value=1000, max_value=999999, value=1000, step=1)
            live_department = st.selectbox("Department", DEPARTMENTS)
            live_student_type = st.selectbox("Student Type", STUDENT_TYPES)
            live_campus_zone = st.selectbox("Campus Zone", CAMPUS_ZONES)
            live_heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=180.0, value=78.0, step=0.1)
            live_systolic = st.number_input("Systolic BP", min_value=80.0, max_value=220.0, value=118.0, step=0.1)
            live_diastolic = st.number_input("Diastolic BP", min_value=50.0, max_value=140.0, value=78.0, step=0.1)
            live_temperature = st.number_input("Temperature (°F)", min_value=94.0, max_value=106.0, value=98.4, step=0.1)
            live_spo2 = st.number_input("SpO2 (%)", min_value=80.0, max_value=100.0, value=97.0, step=0.1)
            live_hydration = st.number_input("Hydration (%)", min_value=0.0, max_value=100.0, value=66.0, step=0.1)
            live_aqi = st.number_input("AQI", min_value=10.0, max_value=500.0, value=90.0, step=0.1)
            live_weather_temp_c = st.number_input("Weather Temp (°C)", min_value=0.0, max_value=50.0, value=29.0, step=0.1)
            live_humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=63.0, step=0.1)
            submitted = st.form_submit_button("Ingest reading")

        if submitted:
            previous_live = (
                st.session_state.live_sensor_df.iloc[-1].to_dict() if not st.session_state.live_sensor_df.empty else None
            )
            normalized_live_record = normalize_ingested_record(
                {
                    "student_id": int(live_student_id),
                    "department": live_department,
                    "student_type": live_student_type,
                    "campus_zone": live_campus_zone,
                    "heart_rate": live_heart_rate,
                    "systolic_bp": live_systolic,
                    "diastolic_bp": live_diastolic,
                    "temperature_f": live_temperature,
                    "spo2": live_spo2,
                    "hydration": live_hydration,
                    "aqi": live_aqi,
                    "weather_temp_c": live_weather_temp_c,
                    "humidity": live_humidity,
                },
                previous_row=previous_live,
            )
            st.session_state.live_sensor_df = pd.concat(
                [st.session_state.live_sensor_df, pd.DataFrame([normalized_live_record])],
                ignore_index=True,
            )
            st.session_state.live_sensor_df = st.session_state.live_sensor_df.tail(rolling_window).reset_index(drop=True)
            st.success("New live sensor reading ingested.")


if data_source == "Simulation":
    latest_previous = st.session_state.health_df.iloc[-1].to_dict() if not st.session_state.health_df.empty else None
    new_record = generate_health_record(previous_row=latest_previous)
    st.session_state.health_df = pd.concat([st.session_state.health_df, pd.DataFrame([new_record])], ignore_index=True)
    st.session_state.health_df = st.session_state.health_df.tail(rolling_window).reset_index(drop=True)
    source_df = st.session_state.health_df.copy()
elif data_source == "CSV Upload":
    source_df = st.session_state.uploaded_df.copy()
else:
    source_df = st.session_state.live_sensor_df.copy()


df = source_df.copy()

if df.empty:
    if data_source == "CSV Upload":
        st.warning("No CSV data available yet. Upload a CSV file from the sidebar.")
    elif data_source == "Live Sensor":
        st.warning("No live sensor readings available yet. Ingest at least one reading from the sidebar form.")
    else:
        st.warning("No data available for current filters. Please change sidebar filters.")
    st.stop()

required_columns = {"timestamp", "department", "student_type"}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    st.error(f"Input data is missing required columns: {', '.join(missing_columns)}")
    st.info("CSV input must include at least: timestamp, department, and student_type.")
    st.stop()

if selected_department != "All":
    df = df[df["department"] == selected_department]
if selected_student_type != "All":
    df = df[df["student_type"] == selected_student_type]

if df.empty:
    if data_source == "CSV Upload":
        st.warning("No CSV data available yet. Upload a CSV file from the sidebar.")
    elif data_source == "Live Sensor":
        st.warning("No live sensor readings available yet. Ingest at least one reading from the sidebar form.")
    else:
        st.warning("No data available for current filters. Please change sidebar filters.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

if df.empty:
    st.warning("No valid timestamp values found in current input data.")
    st.stop()

df = add_zscore_anomaly_flags(
    df,
    cols=["heart_rate", "temperature_f", "spo2", "stress_index", "health_score"],
    threshold=2.0,
)

latest = df.iloc[-1]
risk_color_map = {"Healthy": "green", "Warning": "orange", "Critical": "red"}
risk_label = latest["risk_level"]


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Heart Rate (bpm)", f"{latest['heart_rate']:.1f}")
metric_col2.metric("Blood Pressure (mmHg)", f"{latest['systolic_bp']:.0f}/{latest['diastolic_bp']:.0f}")
metric_col3.metric("Temperature (°F)", f"{latest['temperature_f']:.2f}")
metric_col4.metric("SpO2 (%)", f"{latest['spo2']:.1f}")

if risk_label == "Critical":
    st.error("🚨 Emergency Alert: Critical health pattern detected. Visit Campus Medical Center immediately.")
elif risk_label == "Warning":
    st.warning("⚠️ Warning: Health metrics require attention. Monitor closely and consider medical consultation.")
else:
    st.success("✅ Health status is stable.")

if latest["hydration"] < 45:
    st.info("💧 Hydration Reminder: Current hydration level is low. Please drink water.")

student_baselines = build_student_baselines(source_df)
student_baseline = None
if not student_baselines.empty:
    matching_baseline = student_baselines[student_baselines["student_id"] == latest["student_id"]]
    if not matching_baseline.empty:
        student_baseline = matching_baseline.iloc[0]

personal_severity, personal_findings = personalized_alert_summary(latest, student_baseline)
personal_message = " ".join(personal_findings)
if personal_severity == "error":
    st.error(f"🧠 Personalized Alert (Student {int(latest['student_id'])}): {personal_message}")
elif personal_severity == "warning":
    st.warning(f"🧠 Personalized Alert (Student {int(latest['student_id'])}): {personal_message}")
else:
    st.info(f"🧠 Personalized Insight (Student {int(latest['student_id'])}): {personal_message}")


pred_steps = 5
forecast_values = moving_average_forecast(df[parameter], steps=pred_steps, window=8)
future_time = [df["timestamp"].iloc[-1] + timedelta(seconds=refresh_seconds * (i + 1)) for i in range(pred_steps)]

line_col, gauge_col = st.columns([2, 1])

with line_col:
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[parameter],
            mode="lines+markers",
            name=f"Live {parameter}",
            line=dict(width=2),
        )
    )
    trend_fig.add_trace(
        go.Scatter(
            x=future_time,
            y=forecast_values,
            mode="lines+markers",
            name="Predicted (next 5)",
            line=dict(width=2, dash="dot"),
        )
    )
    anomalies = df[df["is_anomaly"]]
    if not anomalies.empty:
        trend_fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies[parameter],
                mode="markers",
                name="Z-score anomaly",
                marker=dict(size=10, color="red", symbol="x"),
            )
        )
    trend_fig.update_layout(
        title=f"Real-Time {parameter} Trend with Prediction",
        xaxis_title="Time",
        yaxis_title=parameter,
        template="plotly_white",
        legend_title="Series",
        height=420,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

with gauge_col:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(latest["health_score"]),
            title={"text": "Health Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color_map[risk_label]},
                "steps": [
                    {"range": [0, 45], "color": "#ffcccc"},
                    {"range": [45, 70], "color": "#fff2cc"},
                    {"range": [70, 100], "color": "#d9ead3"},
                ],
            },
        )
    )
    gauge.update_layout(height=420, template="plotly_white")
    st.plotly_chart(gauge, use_container_width=True)


risk_text, projected_score = risk_projection_10min(df, refresh_seconds=refresh_seconds)
stress_forecast = rolling_regression_forecast(df["stress_index"], steps=5, window=12)
st.markdown(
    f"**Predictive Insight:** Estimated 10-minute risk outlook is **{risk_text}** (projected score: **{projected_score:.1f}**)."
)
st.caption(f"Stress forecast (next 5 points via rolling regression): {', '.join([f'{x:.1f}' for x in stress_forecast])}")


bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    dept_stress = (
        df.groupby("department", as_index=False)["stress_index"].mean().sort_values("stress_index", ascending=False)
    )
    dept_fig = px.bar(
        dept_stress,
        x="department",
        y="stress_index",
        color="department",
        title="Department-wise Stress Comparison",
        template="plotly_white",
    )
    st.plotly_chart(dept_fig, use_container_width=True)

with bottom_col2:
    scatter = px.scatter(
        df,
        x="aqi",
        y="spo2",
        color="risk_level",
        hover_data=["department", "student_type", "campus_zone"],
        title="AQI Impact on Respiratory Health (SpO2)",
        template="plotly_white",
    )
    st.plotly_chart(scatter, use_container_width=True)


intel_col1, intel_col2 = st.columns(2)

with intel_col1:
    hostel_day = (
        df.groupby(["student_type", "timestamp"], as_index=False)["health_score"].mean().sort_values("timestamp")
    )
    trend_compare = px.line(
        hostel_day,
        x="timestamp",
        y="health_score",
        color="student_type",
        title="Hostel vs Day Scholar Health Trend",
        template="plotly_white",
    )
    st.plotly_chart(trend_compare, use_container_width=True)

with intel_col2:
    heatmap_matrix = build_heatmap_matrix(df)
    if not heatmap_matrix.empty:
        heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_matrix.values,
                x=heatmap_matrix.columns,
                y=heatmap_matrix.index,
                colorscale="RdYlGn",
                reversescale=False,
                colorbar=dict(title="Avg Health Score"),
            )
        )
        heatmap.update_layout(title="Campus Health Status Heatmap", template="plotly_white")
        st.plotly_chart(heatmap, use_container_width=True)


st.subheader("Latest Records")
st.dataframe(df.tail(10), use_container_width=True)

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download current dataset as CSV",
    data=csv_data,
    file_name="christ_smart_healthboard_data.csv",
    mime="text/csv",
)


if data_source == "Simulation" and auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
