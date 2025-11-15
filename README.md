

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
# MIT ROTC Weather AI

## Overview
MIT ROTC Weather AI is a custom long‑range forecasting system built specifically to improve weather intelligence for MIT Air Force ROTC training operations.  
It predicts **rain probability**, **temperature**, and **wind** for the three mission‑critical training windows:

- **Monday PT** — 07:00–08:00  
- **Tuesday LLAB** — 15:00–17:00  
- **Wednesday PT** — 07:00–08:00  

Instead of relying on a single provider (NWS, Weather.com, etc.), this system **learns provider biases**, **models forecast error patterns**, and outputs an AI‑corrected forecast tailored to ROTC needs.

This project is now fully functional end‑to‑end:
- Continuous forecast collection from Open‑Meteo’s 16‑day GFS model  
- Automatic training dataset expansion  
- Neural‑network model training  
- Full prediction engine with ROTC‑specific output  
- Clean unit conversions (°F, mph)  
- Structured weekly/PT/LLAB summaries  

---

# System Architecture (Updated)

The project consists of four major subsystems:

---

## 1. **Data Infrastructure & Environment**
This repo contains:
- Complete Python project with virtual environment
- `training_scripts/` for dataset generation  
- `scripts/` for live predictions
- `models/` directory for saved neural networks
- `data/` directory for accumulated training data

Installed dependencies include:
- `pandas`, `numpy`  
- `requests`  
- `tensorflow` / `keras`  
- `scikit-learn`  
- `matplotlib`  

### Data Sources
We use **Open‑Meteo (GFS model)** because it provides:
- **Hourly forecasts up to 16 days**
- Temperature, precipitation, windspeed, and precipitation probability

This gives complete coverage for:
- PT predictions (7 days ahead)
- LLAB predictions (8–9 days ahead)
- All other long‑range planning

---

## 2. **Data Collection Pipeline**

### a. `fetch_future_forecasts.py`
This script:
- Pulls the **full 16‑day hourly forecast** for MIT
- Appends all results into `main_forecast_training_data.csv`
- Automatically timestamps retrieval times
- Never overwrites prior predictions (builds training history)

Training works because each date accumulates:
- Predictions from 1 day prior  
- Predictions from 2 days prior  
- …up to 15 days prior  

This lets the neural network learn *how forecast accuracy changes with lead time*.

---

### b. `fetch_actual_batch.py`
(Previously: “fetch yesterday’s observations.”)

To simplify training, we now:
- Pull **actual observed weather for many past days at once**
- Apply precipitation labels (`actual_precip_occurred`)
- Apply actual temp + wind
- Merge these into all matching rows across the dataset

This ensures:
- Every stored prediction for a given hour gets matched with the real observed outcome
- Duplicate lead‑time predictions are correctly filled

---

### c. `clean_csv.py`
Ensures dataset integrity:
- Removes duplicate actual columns
- Removes broken merges
- Keeps dataset aligned and clean for training

---

## 3. **Training Pipeline**

### a. `prepare_training_data.py`
This script:
- Loads the entire training CSV  
- Cleans NaNs  
- Builds engineered features:  
  - Hour of day  
  - Day of week  
  - “is PT”?  
  - “is LLAB”?  
  - Lead‑time days  
- Applies scaling to features  
- Builds three training targets:
  - `rain_out` (0/1)
  - `temp_out`
  - `wind_out`

Everything is saved to:
- `feature_cols.json`  
- `scaler.pkl`  
- Ready for neural network training  

---

### b. `train_model.py`
This is the core neural network trainer.

The model:
- Has **three simultaneous outputs**
  - Rain probability (sigmoid → 0–1)
  - Temperature prediction (regression)
  - Wind prediction (regression)
- Uses separate loss functions for each:
  - Binary cross‑entropy  
  - Mean squared error  
  - Mean squared error  
- Trains with masking on missing wind/temp data  
- Saves:
  - `models/weather_nn.keras`  
  - `models/model_info.json`  
  - `models/training_metrics.json`

This completes the machine‑learning stack.

---

## 4. **Prediction Engine (LIVE Forecasting)**

### `predict_weather.py`
This script allows the user to:

#### **1. Input any date**
```
Enter date to predict (YYYY-MM-DD):
```

#### **2. The script automatically:**
- Loads the trained neural network  
- Pulls fresh 16‑day Open‑Meteo forecast  
- Builds engineered features  
- Produces hourly predictions for the selected date  
- Converts output units:
  - °C → °F  
  - km/h → mph  

#### **3. It finds the ROTC week:**
Given *any* entered date:
- Finds the Monday of that week  
- Finds the Tuesday  
- Finds the Wednesday  
- Pulls the prediction rows for:
  - **Monday PT (07:00)**
  - **Tuesday LLAB (15:00, 16:00, 17:00)**
  - **Wednesday PT (07:00)**

#### **4. Output Format**

### Hourly predictions for the selected date:
```
2025-11-18 15:00 | Rain=0.44 | Temp=46.2°F | Wind=12.1 mph
2025-11-18 16:00 | Rain=0.51 | Temp=45.9°F | Wind=11.8 mph
```

### ROTC summary (Option C):
```
➡️ Monday PT (2025-11-17)
➡️ Tuesday LLAB (2025-11-18)
➡️ Wednesday PT (2025-11-19)
```

This gives a complete, AI-enhanced 3‑day operational picture.

---

# Summary of the Build Process (Updated)
1. **Create environment & repo**
2. **Implement future forecast scraping**
3. **Build rolling training dataset (lead‑time aware)**
4. **Add actual weather for all past days**
5. **Engineer features & prepare training matrices**
6. **Train multi‑output neural network**
7. **Save model + scaling logic**
8. **Build full prediction engine**
9. **Generate hourly + ROTC‑specific forecasts**

---

# Technical Stack
- **Python 3.12**
- **TensorFlow / Keras** — neural network  
- **scikit‑learn** — scaling + preprocessing  
- **pandas / numpy** — data management  
- **Open‑Meteo GFS** — 16‑day forecast source  
- **matplotlib** — (optional) visualization  
- **cron / PyCharm Scheduler** — daily runs  

---

# Future Roadmap
- Add precipitation **confidence intervals** (dropout Bayesian)  
- Build a **Streamlit operational dashboard**  
- Add seasonality + bias correction (winter/summer adjustments)  
- Train on multi‑year data  
- Auto‑email/Teams weekly outlook  
- Expand to 30‑day seasonal planning  

---

# Author
Developed by **Harnoor Singh**  
MIT Air Force ROTC  
Contact: `hdsingh@mit.edu`