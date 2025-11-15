import os
import pandas as pd
import requests
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "main_forecast_training_data.csv")

def clear_actual_columns():
    print("🧹 Clearing previous actual columns...")
    df = pd.read_csv(CSV_PATH)
    columns_to_clear = ["actual_temp", "actual_wind", "actual_precip_occurred"]
    for col in columns_to_clear:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    df.to_csv(CSV_PATH, index=False)
    print("✅ Previous actual columns cleared.")

def apply_historical_actuals():
    print("🔎 Loading main forecast file...")
    df = pd.read_csv(CSV_PATH)

    # Build range of dates to fetch
    start = datetime(2025, 11, 9)
    end = datetime(2025, 11, 14)

    dfs_actual_list = []

    print("📡 Fetching hourly actual weather for 2025‑11‑09 → 2025‑11‑14 ...")

    for i in range((end - start).days + 1):
        d = start + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")

        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            "latitude=42.3601&longitude=-71.0589"
            f"&start_date={date_str}&end_date={date_str}"
            "&hourly=temperature_2m,precipitation,windspeed_10m"
            "&timezone=America/New_York"
        )

        r = requests.get(url)
        data = r.json()

        if "hourly" not in data:
            print(f"⚠ No hourly data for {date_str}, skipping…")
            continue

        hourly = data["hourly"]
        df_actual = pd.DataFrame({
            "time": hourly["time"],
            "actual_temp": hourly["temperature_2m"],
            "actual_wind": hourly["windspeed_10m"],
            "actual_precip_occurred": [1 if p > 0 else 0 for p in hourly["precipitation"]],
        })
        dfs_actual_list.append(df_actual)

    if not dfs_actual_list:
        print("❌ No actuals found for any date. Exiting.")
        return

    df_actual_all = pd.concat(dfs_actual_list, ignore_index=True)

    print("🔗 Merging actuals into main dataset...")
    df_merged = df.merge(df_actual_all, on="time", how="left")

    print("💾 Saving updated dataset...")
    df_merged.to_csv(CSV_PATH, index=False)
    print("✅ Historical actuals added successfully!")

# Replace the old main block:
if __name__ == "__main__":
    clear_actual_columns()
    apply_historical_actuals()