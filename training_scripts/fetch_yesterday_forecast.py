#FILE SETUP#
"""
This file pulls forecast from yesterday using Open-Meteo GPS API.
It then saves it to main_forecast_training_data.csv (ony the important parts) including to all duplicates of the same date.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# === SETTINGS ===
LAT, LON = 42.3601, -71.0942  # MIT coordinates
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "main_forecast_training_data.csv")

# === DETERMINE YESTERDAY DATE ===
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.now().strftime("%Y-%m-%d")

print(f"🔄 Fetching hourly data for {yesterday} from Open-Meteo GFS archive...")

# === API REQUEST (PAST WEATHER) ===
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={yesterday}&end_date={yesterday}"
    f"&hourly=precipitation,temperature_2m,windspeed_10m"
    f"&timezone=America/New_York"
)

response = requests.get(url)
data = response.json()

# === PARSE JSON TO DATAFRAME ===
if "hourly" not in data:
    print("⚠️ No hourly data returned for yesterday. Check API or network.")
else:
    hourly = data["hourly"]
    df_actual = pd.DataFrame({
        "time": hourly["time"],
        "precipitation": hourly["precipitation"],
        "temperature_2m": hourly["temperature_2m"],
        "windspeed_10m": hourly["windspeed_10m"],
    })
    df_actual["actual_precip_occurred"] = (df_actual["precipitation"] > 0).astype(int)
    df_actual = df_actual[["time", "actual_precip_occurred", "temperature_2m", "windspeed_10m"]]
    df_actual.rename(columns={"temperature_2m": "actual_temp", "windspeed_10m": "actual_wind"}, inplace=True)

    # === ENSURE DATA FOLDER EXISTS ===
    os.makedirs(DATA_DIR, exist_ok=True)

    # === READ MAIN FORECAST FILE ===
    if not os.path.exists(CSV_PATH):
        print(f"⚠️ Main forecast file not found at {CSV_PATH}. Cannot update.")
    else:
        df_main = pd.read_csv(CSV_PATH)
        # === DROP OLD ACTUAL COLUMNS TO PREVENT MERGE COLLISIONS ===
        cols_to_drop = [
            "actual_precip_occurred",
            "actual_temp",
            "actual_wind"
        ]
        df_main = df_main.drop(columns=[c for c in cols_to_drop if c in df_main.columns], errors="ignore")
        # Check if any entries for yesterday exist in main dataframe
        mask = df_main["time"].str.startswith(yesterday)
        if not mask.any():
            print("No matching forecast entries for yesterday found. Skipping update.")
        else:
            # Merge actual data into main dataframe on 'time'
            # For multiple entries with same time, all will be updated
            df_main = df_main.merge(df_actual, on="time", how="left")

            # Save updated dataframe
            df_main.to_csv(CSV_PATH, index=False)
            print(f"✅ Updated main forecast training data with actuals for {yesterday} at {CSV_PATH}")
            print(f"📊 Total records updated: {mask.sum()}")
