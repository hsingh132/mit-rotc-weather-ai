

# MIT ROTC Weather AI

## Overview
MIT ROTC Weather AI is a custom forecasting and data intelligence system built to improve the accuracy of weather forecasts for MIT Air Force ROTC training activities. It focuses on three specific time windows:
- **Monday PT** (7–8 AM)
- **Wednesday PT** (7–8 AM)
- **Tuesday LLAB** (3–5 PM)

The system’s goal is to tell whether rain or poor conditions are most likely *inside or outside* those exact training windows, with higher accuracy than single-source forecasts.

---

## Project Phases (4 Repositories / Core Components)

### 1. Environment & Data Infrastructure
This is where the foundation of the project is built.
- Set up a GitHub repo, Python virtual environment, and directory structure.
- Installed libraries: `pandas`, `numpy`, `SQLAlchemy`, `requests`, `xgboost`, `scikit-learn`, `torch`, etc.
- Created a local SQLite database (`weather_data.db`) to store forecasts and observed data.
- Defined the schema with two tables:
  - `forecasts` – for weather provider predictions
  - `observations` – for real-world recorded weather

**Purpose:** This phase ensures all data has a consistent home and can be queried easily for model training later.

---

### 2. Data Collection Pipeline
This phase connects to APIs and downloads weather data automatically.

#### a. Forecast Data
- Script: `fetch_forecasts.py`
- Uses NOAA’s NWS API to pull hourly forecasts for MIT’s location.
- Stores provider name, forecast timestamp, target time, temperature, precipitation chance, wind, etc.
- Each entry includes a “lead time” (hours ahead of target time).

#### b. Observation Data (Ground Truth)
- Script: `fetch_observations.py`
- Uses NOAA’s open observations API (`api.weather.gov/stations/KBOS/observations`) to get actual measured conditions.
- Records temperature, wind, gusts, and precipitation flags.
- These represent the *truth* that models will learn from.

**Purpose:** Build a complete, continually updating dataset of both predictions and actual outcomes.

---

### 3. Data Processing & Model Training
Once forecasts and observations are collected, this phase aligns and analyzes them.

- Match forecasts to actual outcomes based on time (e.g., what each provider predicted vs. what occurred).
- Label outcomes for ROTC-specific windows:
  - Rain **IN** LLAB/PT window → `1`
  - Rain **OUT** of window → `0`
- Train a machine learning model:
  - Version 0.1: Logistic regression or XGBoost (baseline).
  - Version 1.0: Neural network (multi-input, sequence-based).
- Evaluate performance:
  - Accuracy, F1 score, and comparison to each provider.

**Purpose:** Teach the system how to predict rain probability and confidence specifically for ROTC schedules.

---

### 4. Reporting & Automation
This phase turns model output into readable reports and automates delivery.

- Generates a forecast card in a table format:
  ```
  | Day | Description | High | Low | Rain % | Wind/Gust | ROTC Window Note | Confidence |
  |-----|--------------|------|------|--------|------------|------------------|-------------|
  | Tue | Cloudy | 68°F | 54°F | 60% | 8/14 | Rain likely IN LLAB (3–5 PM) | 0.82 |
  ```
- Integrates with Microsoft Teams or email to auto-post weekly reports.
- Can later include dashboard visualizations (e.g., Streamlit app).

**Purpose:** Deliver actionable, time-specific weather intelligence directly to MIT ROTC leadership.

---

## Summary of the Build Process
1. **Set up the environment** – Create repo, virtual environment, install dependencies.
2. **Define the database schema** – Create `forecasts` and `observations` tables.
3. **Implement forecast scraping** – Collect hourly forecasts from NOAA/NWS.
4. **Implement observation fetching** – Collect actual observed weather (KBOS station).
5. **Process & clean data** – Match forecast vs. observed data.
6. **Train models** – Build predictive models for 7-day and 10-day lead windows.
7. **Generate reports** – Output “rain likely IN/OUT” summaries.
8. **Automate updates** – Schedule daily fetches and weekly report generation.

---

## Technical Stack
- **Python 3.12**
- **SQLite / SQLAlchemy** – database storage
- **pandas, numpy** – data manipulation
- **requests** – API communication
- **scikit-learn / XGBoost / PyTorch** – machine learning
- **matplotlib** – visualizations
- **cron / GitHub Actions** – automation

---

## Future Roadmap
- Add confidence metrics and seasonal bias adjustments.
- Deploy a small web dashboard.
- Integrate with MIT Teams for automatic weather summaries.
- Expand model to 14-day planning horizon.

---

## Author
Developed by **Harnoor Singh** as part of MIT Air Force ROTC innovation initiatives.
Contact: `hdsingh@mit.edu`