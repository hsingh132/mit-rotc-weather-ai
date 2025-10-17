import requests
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from setup_database import Observation
import os

# --- DB setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "data", "weather_data.db")
engine = create_engine(f"sqlite:///{db_path}")
Session = sessionmaker(bind=engine)
session = Session()

# --- Parameters ---
station = "KBOS"  # Logan Airport
limit = 100  # last ~100 observations (roughly 24-36h depending on cadence)

url = f"https://api.weather.gov/stations/{station}/observations?limit={limit}"
headers = {
    "User-Agent": "MIT-ROTC-Weather-App (harnoorsingh@mit.edu)",
    "Accept": "application/geo+json"
}

print(f"Fetching latest observations for {station} from api.weather.gov ...")
resp = requests.get(url, headers=headers, timeout=20)
resp.raise_for_status()
data = resp.json()

features = data.get("features", [])
print(f"Fetched {len(features)} observations")

def to_float(v):
    try:
        return float(v)
    except Exception:
        return None

def c_to_f(c):
    return c * 9.0/5.0 + 32.0 if c is not None else None

def ms_to_mph(ms):
    return ms * 2.236936 if ms is not None else None

count = 0
for feat in features:
    try:
        props = feat.get("properties", {})
        # timestamp like "2025-10-16T22:52:00+00:00"
        tstr = props.get("timestamp")
        if not tstr:
            continue
        ts = datetime.fromisoformat(tstr.replace("Z", "+00:00"))

        temp_c = props.get("temperature", {}).get("value")  # in °C
        wind_ms = props.get("windSpeed", {}).get("value")   # in m/s
        gust_ms = props.get("windGust", {}).get("value")    # in m/s
        # precipitationLastHour in mm per docs (may be null)
        precip_mm = props.get("precipitationLastHour", {}).get("value")

        temp_f = c_to_f(to_float(temp_c))
        wind_mph = ms_to_mph(to_float(wind_ms))
        gust_mph = ms_to_mph(to_float(gust_ms))

        precip_occurred = False
        if precip_mm is not None:
            try:
                precip_occurred = float(precip_mm) > 0.0
            except Exception:
                precip_occurred = False
        else:
            # Fallback: check presentWeather codes or rawMessage for RA/SN/DZ/FG/BR
            wx = props.get("presentWeather") or []
            text_join = " ".join([w.get("intensity", "") + w.get("weather", "") for w in wx])
            raw = props.get("rawMessage") or ""
            if any(code in (text_join + " " + raw) for code in [" RA", " SN", " DZ", " FG", " BR", "RA", "SN", "DZ", "FG", "BR"]):
                precip_occurred = True

        obs = Observation(
            timestamp=ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts,
            station=station,
            temp_f=temp_f,
            wind_mph=wind_mph,
            gust_mph=gust_mph,
            precip_occurred=precip_occurred,
        )
        session.add(obs)
        count += 1
    except Exception as e:
        print("Skipping observation due to error:", e)
        continue

session.commit()
print(f"✅ Inserted {count} KBOS observations into database.")