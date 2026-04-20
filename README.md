# 🩺 SmartHealthAnalyzer

A real-time campus health intelligence dashboard built for CHRIST University. It streams student vital signs, detects anomalies using Z-score analysis, predicts near-future health trends with moving average and rolling regression models, and delivers personalized alerts based on each student's individual baseline — all in an interactive Streamlit interface.

---

## What It Does

Campus health monitoring traditionally relies on manual check-ups and reactive responses. SmartHealthAnalyzer replaces that with a continuous, data-driven pipeline that:

- Simulates live student health data with realistic physiological correlations (department stress shifts, campus zone AQI effects, weather-temperature coupling)
- Detects statistical anomalies in real time using Z-score flagging across five vital parameters
- Predicts the next 5 data points using a moving average forecast and projects a 10-minute risk outlook using linear regression
- Compares each student's current vitals against their own historical baseline (10th–90th percentile range), not just global thresholds
- Visualizes department-level stress comparisons, AQI impact on SpO₂, hostel vs day scholar health trends, and a campus zone heatmap

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard & UI | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualizations | Plotly Express, Plotly Graph Objects |
| Analytical Models | Custom implementations in `analysis.py` |
| Health Scoring | Weighted multi-parameter formula in `utils.py` |
| Language | Python 3.10+ |

---

## Features

### Three Data Input Modes
- **Simulation** — Auto-generates a new health record every 2–3 seconds with continuity from the previous record. Runs indefinitely with auto-refresh.
- **CSV Upload** — Upload any CSV with health data. The app normalizes and validates every row, computing derived metrics (stress index, health score, risk level) from raw vitals.
- **Live Sensor** — Manually ingest individual sensor readings via a sidebar form. Each reading is normalized and appended to a rolling window.

### Health Scoring Engine
Every record is scored using two computed metrics:

**Stress Index** — Weighted combination of heart rate deviation, blood pressure deviation, SpO₂ deficit, hydration deficit, and AQI level. Each component is clamped and scaled to 0–100.

**Health Score** — Weighted abnormality score across heart rate, blood pressure, temperature, SpO₂, and stress index. Higher score = healthier. Thresholds: `≥ 70` → Healthy, `45–69` → Warning, `< 45` → Critical.

### Anomaly Detection
Z-score flagging is applied across five parameters — heart rate, temperature, SpO₂, stress index, and health score. Any record with `|z| ≥ 2.0` on at least one parameter is flagged as an anomaly and highlighted on the trend chart with a red ✕ marker.

### Predictive Analytics
- **Moving Average Forecast** — Uses the last 8 values to predict the next 5 data points for any selected health parameter.
- **Rolling Regression Forecast** — Fits a linear regression over the last 12 stress index values and projects the next 5 points.
- **10-Minute Risk Projection** — Extrapolates the health score trend using linear regression over the last 25 records, calculates the projected score at `t + 600 seconds`, and classifies it as Stable, Moderate Risk, or High Risk.

### Personalized Baseline Alerts
For each student with 5 or more records in the session, the app computes per-student 10th, 50th, and 90th percentile baselines for heart rate, SpO₂, and stress index. Current vitals outside the personal 10th–90th percentile range trigger a personalized warning — surfacing deviations that global thresholds would miss.

### Dashboard Visualizations
- Real-time line chart with predicted trend overlay and anomaly markers
- Health risk gauge (0–100 score with color-coded risk bands)
- Department-wise average stress bar chart
- AQI vs SpO₂ scatter plot colored by risk level
- Hostel vs Day Scholar health score trend comparison
- Campus zone × department health score heatmap

---

## Project Structure

```
SmartHealthAnalyzer/
├── app.py                   # Streamlit dashboard — UI, data flow, chart rendering
├── analysis.py              # Analytical models — forecasting, anomaly detection, risk projection
├── utils.py                 # Health scoring engine, data generation, normalization, baselines
├── sample_health_data.csv   # Sample CSV for testing the upload mode
└── requirements.txt         # Python dependencies
```

### Module Responsibilities

**`utils.py`**
- `generate_health_record()` — Produces a physiologically correlated synthetic health record. Applies department stress shifts, campus zone AQI offsets, weather-temperature coupling, and continuity from the previous record.
- `calculate_stress_index()` — Weighted formula across HR, BP, SpO₂, hydration, and AQI.
- `calculate_health_score()` — Weighted abnormality scoring across five vitals.
- `normalize_ingested_record()` — Validates and clamps any externally ingested record, recomputes derived metrics.
- `build_student_baselines()` — Computes per-student percentile baselines from session history.
- `personalized_alert_summary()` — Compares latest reading to personal baseline and returns severity + findings.

**`analysis.py`**
- `moving_average_forecast()` — Iterative window-mean forecast for any numeric series.
- `rolling_regression_forecast()` — Linear regression over recent window, extrapolates forward with clipping.
- `add_zscore_anomaly_flags()` — Adds per-column Z-score anomaly flags and an aggregate `is_anomaly` boolean.
- `risk_projection_10min()` — Linear regression on recent health scores, projects to `t + 10 minutes`.
- `build_heatmap_matrix()` — Pivots department × campus zone mean health scores for the heatmap.

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/sharonmattheww/SmartHealthAnalyzer.git
cd SmartHealthAnalyzer
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Using the Dashboard

**Simulation mode** starts automatically. The dashboard seeds 80 records on first load and appends a new record every 2–3 seconds. Use the sidebar to:
- Switch between Simulation, CSV Upload, and Live Sensor modes
- Filter by department (Engineering, Commerce, Arts) and student type (Hostel, Day Scholar)
- Select the health parameter to visualize on the trend chart
- Adjust the rolling window size (50–100 records)
- Toggle auto-refresh on or off

**CSV Upload mode** — Use `sample_health_data.csv` as a reference for the expected column format. Required columns: `timestamp`, `department`, `student_type`. All other columns are normalized and validated on load.

**Live Sensor mode** — Fill in the sidebar form with a student's vitals and click "Ingest reading." The record is normalized, scored, and appended to the live session dataset.

---

## Health Parameters Tracked

| Parameter | Unit | Description |
|---|---|---|
| `heart_rate` | bpm | Resting heart rate |
| `systolic_bp` | mmHg | Systolic blood pressure |
| `diastolic_bp` | mmHg | Diastolic blood pressure |
| `temperature_f` | °F | Body temperature |
| `spo2` | % | Blood oxygen saturation |
| `hydration` | % | Hydration level |
| `aqi` | AQI index | Ambient air quality at campus zone |
| `weather_temp_c` | °C | Outdoor temperature |
| `humidity` | % | Relative humidity |
| `stress_index` | 0–100 | Computed composite stress score |
| `health_score` | 0–100 | Computed overall health score |
| `risk_level` | Healthy / Warning / Critical | Derived from health score |

---

## Sample Data

`sample_health_data.csv` is included in the repository to test the CSV upload mode. It follows the same schema as the simulated data and can be used to verify normalization, scoring, and chart rendering without running in simulation mode.
