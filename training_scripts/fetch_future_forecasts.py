#FILE SETUP#
"""
This file pulls forecasts from 5 days in advance to 15 days in advance using Open-Meteo GPS API.
It then saves it to main_forecast_training_data.csv.
"""

# === IMPORTS ===
import os
import requests
import pandas as pd
from datetime import datetime

# === SETTINGS ===
LAT, LON = 42.3601, -71.0942  # MIT coordinates
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "main_forecast_training_data.csv")

# === API REQUEST ===
print("🔄 Fetching 16-day hourly forecast data from Open-Meteo (GFS)...")
url = (
    f"https://api.open-meteo.com/v1/gfs?"
    f"latitude={LAT}&longitude={LON}"
    f"&hourly=precipitation_probability,precipitation,temperature_2m,windspeed_10m"
    f"&forecast_days=16&timezone=America/New_York"
)
response = requests.get(url)
data = response.json()

# === PARSE JSON TO DATAFRAME ===
hourly = data["hourly"]
df = pd.DataFrame(hourly)
df["retrieved_at"] = datetime.now().isoformat()

print(f"✅ Retrieved {len(df)} hourly records from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

# === ENSURE DATA FOLDER EXISTS ===
os.makedirs(DATA_DIR, exist_ok=True)

# === SAVE OR APPEND TO CSV ===
if not os.path.exists(CSV_PATH):
    df.to_csv(CSV_PATH, index=False)
    print(f"📁 Created new CSV file: {CSV_PATH}")
else:
    df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    print(f"📈 Appended {len(df)} new rows to {CSV_PATH}")

print("✅ Forecast data successfully saved!")
