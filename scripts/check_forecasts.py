from sqlalchemy import create_engine, text
import os

# Build absolute path to the database
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "data", "weather_data.db")

# Connect to your database
engine = create_engine(f"sqlite:///{db_path}")

with engine.connect() as conn:
    # 1️⃣ Check how many total rows you have
    result = conn.execute(text("SELECT COUNT(*) FROM forecasts"))
    print("Total forecasts:", result.scalar())

    # 2️⃣ See the first few entries
    rows = conn.execute(text("""
        SELECT provider, date, target_time, precip_prob, temp_f, wind_mph, lead_time_hr
        FROM forecasts
        LIMIT 5
    """))
    for row in rows:
        print(row)