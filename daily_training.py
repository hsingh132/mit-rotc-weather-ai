#FILE SETUP#
"""
daily_training.py
This is the control panel that runs both forecast-fetching training_scripts in sequence.
"""

import subprocess
import os

# === Define paths ===
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "training_scripts")
fetch_future_path = os.path.join(SCRIPTS_DIR, "fetch_future_forecasts.py")
fetch_yesterday_path = os.path.join(SCRIPTS_DIR, "fetch_yesterday_forecast.py")
# === Run training_scripts sequentially ===
print("🚀 Starting daily forecast update process...")

print("\n[1/2] Running fetch_future_forecasts.py...")
subprocess.run(["python3", fetch_future_path], check=True)

# print("\n[2/2] Running fetch_yesterday_forecast.py...")
# subprocess.run(["python3", fetch_yesterday_path], check=True)
# (Temporarily disabled: actuals will be added later)

print("\n✅ Daily training data update complete!")
