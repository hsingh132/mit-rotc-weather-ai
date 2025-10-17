from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker  # modern declarative base import
import os

# Always use absolute path from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "data", "weather_data.db")

# 1. Define base and engine
Base = declarative_base()
engine = create_engine(f"sqlite:///{db_path}")  # local file database

# 2. Forecasts table
class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True)
    date = Column(Date)                      # Date being forecasted
    provider = Column(String)                # NWS, Weather.com, etc.
    forecast_time = Column(DateTime)         # When forecast was issued
    target_time = Column(DateTime)           # The actual time being predicted
    lead_time_hr = Column(Integer)           # Hours ahead
    precip_prob = Column(Float)              # %
    temp_f = Column(Float)
    wind_mph = Column(Float)
    gust_mph = Column(Float)

# 3. Observations table
class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    station = Column(String)                 # e.g. KBOS
    precip_occurred = Column(Boolean)        # True/False
    temp_f = Column(Float)
    wind_mph = Column(Float)
    gust_mph = Column(Float)

# 4. Create tables
Base.metadata.create_all(engine)

print("✅ Database schema created successfully!")