

"""
clean_csv.py
This script cleans the main_forecast_training_data.csv file by:
- Removing ALL duplicate actual_* columns (including _x, _y versions)
- Keeping only one set: actual_precip_occurred, actual_temp, actual_wind
- Ensuring the CSV is consistent moving forward
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "main_forecast_training_data.csv")

print("🔧 Cleaning main_forecast_training_data.csv ...")

# Load CSV
df = pd.read_csv(DATA_PATH)

# Identify all actual-related columns
actual_cols = [c for c in df.columns if c.startswith("actual_")]

print(f"Found actual-related columns: {actual_cols}")

# Extract preserved values (prefer non-null in priority order)
def merge_actual_column(prefix):
    """Merge all columns starting with prefix into one."""
    cols = [c for c in actual_cols if c.startswith(prefix)]
    if not cols:
        return None

    series = df[cols].bfill(axis=1).ffill(axis=1).iloc[:, 0]
    return series

merged_actual_precip = merge_actual_column("actual_precip_occurred")
merged_actual_temp   = merge_actual_column("actual_temp")
merged_actual_wind   = merge_actual_column("actual_wind")

# Drop all actual columns
df = df.drop(columns=actual_cols, errors="ignore")

# Add back the clean, merged ones
df["actual_precip_occurred"] = merged_actual_precip
df["actual_temp"] = merged_actual_temp
df["actual_wind"] = merged_actual_wind

# Save cleaned CSV
df.to_csv(DATA_PATH, index=False)

print("✅ Cleaning complete!")
print("Saved cleaned file to:", DATA_PATH)