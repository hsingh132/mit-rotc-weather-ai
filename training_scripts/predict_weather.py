import os
import json
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from tensorflow import keras

# ============================================================
#  CONFIG / PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "weather_nn.keras")
FEATURE_COLS_PATH = os.path.join(DATA_DIR, "feature_cols.json")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

# Location (MIT Campus)
LAT = 42.3601
LON = -71.0942

# ============================================================
#  STEP 1 — Load model + metadata
# ============================================================

print("📂 Loading model + scaler...")

model = keras.models.load_model(MODEL_PATH)

with open(FEATURE_COLS_PATH, "r") as f:
    feature_cols = json.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ============================================================
#  STEP 2 — Ask user for date
# ============================================================

input_date = input("Enter date to predict (YYYY-MM-DD): ").strip()

try:
    datetime.strptime(input_date, "%Y-%m-%d")
except ValueError:
    raise ValueError("❌ Invalid date format. Use YYYY-MM-DD.")

print(f"\n📅 Generating forecast for: {input_date}\n")

# ============================================================
#  STEP 3 — Pull 16-day forecast from Open-Meteo
# ============================================================

print("🔄 Fetching future forecast data from Open-Meteo...")

URL = (
    "https://api.open-meteo.com/v1/forecast?"
    f"latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,precipitation,precipitation_probability,windspeed_10m"
    "&forecast_days=16"
)

resp = requests.get(URL)
if resp.status_code != 200:
    raise RuntimeError("❌ Failed to fetch forecast data from Open-Meteo.")

data = resp.json()

df_all = pd.DataFrame({
    "time": data["hourly"]["time"],
    "temperature_2m": data["hourly"]["temperature_2m"],
    "precipitation": data["hourly"]["precipitation"],
    "precipitation_probability": data["hourly"]["precipitation_probability"],
    "windspeed_10m": data["hourly"]["windspeed_10m"],
})

df_day = df_all[df_all["time"].str.startswith(input_date)]

if df_day.empty:
    raise ValueError("❌ Forecast not available for this date (beyond 16-day range).")

print(f"✅ Retrieved {len(df_day)} hourly rows.\n")

# ============================================================
#  STEP 4 — Build model-ready input features
# ============================================================

print("🛠 Preparing model input features...")

df_all["retrieved_at"] = datetime.utcnow().isoformat()

# Build engineered features required by the model
df_all["lead_time_days"] = 0  # For predictions, assume lead-time = 0
df_all["time_dt"] = pd.to_datetime(df_all["time"])
df_all["hour_of_day"] = df_all["time_dt"].dt.hour
df_all["day_of_week"] = df_all["time_dt"].dt.weekday
df_all["is_pt_window"] = df_all["hour_of_day"].apply(lambda h: 1 if h == 7 else 0)
df_all["is_llab_window"] = df_all["hour_of_day"].apply(lambda h: 1 if h in [15, 16, 17] else 0)

# Drop helper
df_all = df_all.drop(columns=["time_dt"])

# Now confirm all feature columns exist
missing_cols = [c for c in feature_cols if c not in df_all.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required feature columns: {missing_cols}")

df_features = df_all[feature_cols].copy()

X = scaler.transform(df_features.values)

# ============================================================
#  STEP 5 — Run predictions
# ============================================================

print("🤖 Running predictions...\n")

preds = model.predict(X)

rain_pred = preds["rain_out"].flatten()
temp_pred = preds["temp_out"].flatten()
wind_pred = preds["wind_out"].flatten()

results_all = pd.DataFrame({
    "time": df_all["time"],
    "rain_prob": rain_pred,
    "temp_pred_C": temp_pred,
    "wind_pred_m_s": wind_pred
})

# Convert temperature to Fahrenheit and wind speed to mph
results_all["temp_pred_F"] = results_all["temp_pred_C"] * 9/5 + 32
results_all["wind_pred_mph"] = results_all["wind_pred_m_s"] * 2.23694

results_day = results_all[results_all["time"].str.startswith(input_date)]

# ============================================================
#  STEP 6 — Output full table
# ============================================================

print("===============================================")
print("📊 HOURLY FORECAST PREDICTIONS")
print("===============================================")

for _, row in results_day.iterrows():
    dt = datetime.strptime(row['time'], "%Y-%m-%dT%H:%M")
    time_str = dt.strftime("%Y-%m-%d %H:%M")
    print(
        f"{time_str} | "
        f"Rain={row['rain_prob']:.2f} | "
        f"Temp={row['temp_pred_F']:.1f}°F | "
        f"Wind={row['wind_pred_mph']:.1f} mph"
    )

# ============================================================
#  STEP 7 — ROTC TRAINING WINDOWS (Option C)
# ============================================================

print("\n\n===============================================")
print("🎯 ROTC TRAINING WINDOWS (AUTO WEEK MODE)")
print("===============================================")

def print_window(df_slice, label):
    if df_slice.empty:
        print(f"{label}: No data (outside 16-day range?)")
        return
    print(f"\n➡️ {label}")
    for _, row in df_slice.iterrows():
        dt = datetime.strptime(row['time'], "%Y-%m-%dT%H:%M")
        time_str = dt.strftime("%Y-%m-%d %H:%M")
        print(
            f"{time_str} | "
            f"Rain={row['rain_prob']:.2f} | "
            f"Temp={row['temp_pred_F']:.1f}°F | "
            f"Wind={row['wind_pred_mph']:.1f} mph"
        )

input_dt = datetime.strptime(input_date, "%Y-%m-%d")
week_monday = input_dt - timedelta(days=input_dt.weekday())
week_tuesday = week_monday + timedelta(days=1)
week_wednesday = week_monday + timedelta(days=2)

def extract_day(df, day_dt, hours):
    day_str = day_dt.strftime("%Y-%m-%d")
    mask = df["time"].str.startswith(day_str) & df["time"].str.contains(hours)
    return df[mask]

# Monday PT (07:00)
df_mon_pt = extract_day(results_all, week_monday, "T07:00")
print_window(df_mon_pt, f"Monday PT ({week_monday.date()})")

# Tuesday LLAB (15:00–17:00)
df_tue_llab = results_all[
    results_all["time"].str.startswith(week_tuesday.strftime("%Y-%m-%d")) &
    results_all["time"].str.contains("T15:00|T16:00|T17:00")
]
print_window(df_tue_llab, f"Tuesday LLAB ({week_tuesday.date()})")

# Wednesday PT (07:00)
df_wed_pt = extract_day(results_all, week_wednesday, "T07:00")
print_window(df_wed_pt, f"Wednesday PT ({week_wednesday.date()})")

print("\n✅ DONE.")