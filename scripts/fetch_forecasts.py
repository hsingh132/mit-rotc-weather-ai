import requests
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from setup_database import Forecast
import os

# --- DB setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "data", "weather_data.db")
engine = create_engine(f"sqlite:///{db_path}")
Session = sessionmaker(bind=engine)
session = Session()

# --- Step 1: Get grid forecast links from the point endpoint ---
point_url = "https://api.weather.gov/points/42.3601,-71.0942"
point_resp = requests.get(point_url, headers={"User-Agent": "MIT-ROTC-Weather-App"})
point_resp.raise_for_status()
point_data = point_resp.json()

# --- Step 2: Follow the link to the hourly forecast endpoint ---
forecast_url = point_data["properties"]["forecastHourly"]
response = requests.get(forecast_url, headers={"User-Agent": "MIT-ROTC-Weather-App"})
response.raise_for_status()
data = response.json()
periods = data["properties"]["periods"]

for p in periods:
    # Convert and calculate lead time (hours ahead)
    target_dt = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
    lead_time_hr = int((target_dt - datetime.now(timezone.utc)).total_seconds() / 3600)

    forecast = Forecast(
        date=target_dt.date(),
        provider="NWS",
        forecast_time=datetime.now(timezone.utc),     # timezone-aware timestamp
        target_time=target_dt,
        lead_time_hr=lead_time_hr,
        precip_prob=p["probabilityOfPrecipitation"]["value"] or 0,
        temp_f=p["temperature"],
        wind_mph=float(p["windSpeed"].split()[0]) if p["windSpeed"] else None,
        gust_mph=None
    )
    session.add(forecast)

session.commit()
print(f"✅ Inserted {len(periods)} NWS hourly forecasts into the database.")