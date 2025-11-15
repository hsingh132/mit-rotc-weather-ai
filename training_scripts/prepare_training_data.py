import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv("data/main_forecast_training_data.csv")

# Rename and parse datetimes
df['forecast_datetime'] = pd.to_datetime(df['retrieved_at'], errors='coerce')
df['target_datetime'] = pd.to_datetime(df['time'], errors='coerce')

# -------------------------
# 2. Feature Engineering
# -------------------------

# Lead time (days)
df['lead_time_days'] = (df['target_datetime'] - df['forecast_datetime']).dt.total_seconds() / 86400

# Hour of day & day of week
df['hour_of_day'] = df['target_datetime'].dt.hour
df['day_of_week'] = df['target_datetime'].dt.dayofweek   # Monday=0

# PT and LLAB windows
df['is_pt_window'] = 0
df['is_llab_window'] = 0

# PT: Monday or Wednesday at 07:00
df.loc[
    (df['day_of_week'].isin([0, 2])) & (df['hour_of_day'] == 7),
    'is_pt_window'
] = 1

# LLAB: Tuesday at 15:00 or 16:00
df.loc[
    (df['day_of_week'] == 1) & (df['hour_of_day'].isin([15, 16])),
    'is_llab_window'
] = 1

# -------------------------
# 3. Targets
# -------------------------
target_cols = ['actual_precip_occurred', 'actual_temp', 'actual_wind']

# Drop rows missing any actual values
df = df.dropna(subset=target_cols)

# -------------------------
# 4. Feature Columns
# -------------------------
feature_cols = [
    'precipitation_probability',
    'precipitation',
    'temperature_2m',
    'windspeed_10m',
    'lead_time_days',
    'hour_of_day',
    'day_of_week',
    'is_pt_window',
    'is_llab_window'
]

# Ensure numeric
for col in feature_cols + target_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=feature_cols + target_cols)

# -------------------------
# 5. Time-based Split
# -------------------------
df = df.sort_values('forecast_datetime').reset_index(drop=True)
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X = df[feature_cols].values
y = df[target_cols].values

X_train = X[:train_end]
X_val   = X[train_end:val_end]
X_test  = X[val_end:]

y_train = y[:train_end]
y_val   = y[train_end:val_end]
y_test  = y[val_end:]

# -------------------------
# 6. Scale Features
# -------------------------
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# -------------------------
# 7. Save Outputs
# -------------------------
np.save("data/X_train.npy", X_train_scaled)
np.save("data/X_val.npy", X_val_scaled)
np.save("data/X_test.npy", X_test_scaled)

np.save("data/y_train.npy", y_train)
np.save("data/y_val.npy", y_val)
np.save("data/y_test.npy", y_test)

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("data/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

print("✅ Training data prepared successfully!")
print(f"Total samples: {n}")
print(f"Train: {len(X_train_scaled)}, Val: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")
print("Features:", feature_cols)
