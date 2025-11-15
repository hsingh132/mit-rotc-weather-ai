"""
train_model.py
Full multi-output neural network training for:
 - Rain occurrence (classification)
 - Temperature (regression)
 - Wind speed (regression)

Inputs are prepared by prepare_training_data.py and saved in /data.
Artifacts produced here are saved into /models.
"""

import os
import json
import numpy as np
from datetime import datetime

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sklearn utilities
from sklearn.utils.class_weight import compute_class_weight

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------
# 1) Load prepared artifacts
# ---------------------------
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

with open(os.path.join(DATA_DIR, "feature_cols.json"), "r") as f:
    feature_columns = json.load(f)

# y arrays are expected to have columns in this order:
# [actual_precip_occurred (0/1), actual_temp (°C), actual_wind (m/s)]
num_targets = y_train.shape[1]
if num_targets < 1:
    raise ValueError("Targets array has no columns. Ensure prepare_training_data.py produced targets.")

# Slice targets
y_rain_train = y_train[:, 0] if num_targets >= 1 else None
y_temp_train = y_train[:, 1] if num_targets >= 2 else None
y_wind_train = y_train[:, 2] if num_targets >= 3 else None

y_rain_val = y_val[:, 0] if num_targets >= 1 else None
y_temp_val = y_val[:, 1] if num_targets >= 2 else None
y_wind_val = y_val[:, 2] if num_targets >= 3 else None

y_rain_test = y_test[:, 0] if num_targets >= 1 else None
y_temp_test = y_test[:, 1] if num_targets >= 2 else None
y_wind_test = y_test[:, 2] if num_targets >= 3 else None

# For convenience, define X as full set only for printing shapes
X = np.concatenate([X_train, X_val, X_test], axis=0)

print("✅ Loaded data:")
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
print(f"  Features: {len(feature_columns)}")
print(f"  Targets present: rain={y_rain_train is not None}, temp={y_temp_train is not None}, wind={y_wind_train is not None}")

# -----------------------------------------
# 2) Build multi-output Keras architecture
# -----------------------------------------
inputs = layers.Input(shape=(X.shape[1],), name="features")

# Shared trunk
x = layers.Dense(128, activation="relu")(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)

# Heads (use only if targets exist)
outputs = {}
losses = {}
metrics = {}
loss_weights = {}

if y_rain_train is not None:
    rain_head = layers.Dense(32, activation="relu")(x)
    rain_out = layers.Dense(1, activation="sigmoid", name="rain_out")(rain_head)
    outputs["rain_out"] = rain_out
    losses["rain_out"] = "binary_crossentropy"
    metrics["rain_out"] = [keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="prec"), keras.metrics.Recall(name="rec")]
    loss_weights["rain_out"] = 1.0

if y_temp_train is not None:
    temp_head = layers.Dense(32, activation="relu")(x)
    temp_out = layers.Dense(1, activation="linear", name="temp_out")(temp_head)
    outputs["temp_out"] = temp_out
    losses["temp_out"] = "mse"
    metrics["temp_out"] = [keras.metrics.RootMeanSquaredError(name="rmse")]
    loss_weights["temp_out"] = 0.5

if y_wind_train is not None:
    wind_head = layers.Dense(32, activation="relu")(x)
    wind_out = layers.Dense(1, activation="linear", name="wind_out")(wind_head)
    outputs["wind_out"] = wind_out
    losses["wind_out"] = "mse"
    metrics["wind_out"] = [keras.metrics.RootMeanSquaredError(name="rmse")]
    loss_weights["wind_out"] = 0.5

if not outputs:
    raise ValueError("No targets present to train on. Ensure actual labels exist in the dataset.")

model = keras.Model(inputs=inputs, outputs=outputs, name="weather_multihead_nn")

# -----------------------------------------
# 3) Compile with optimizer & callbacks
# -----------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODELS_DIR, "checkpoint.keras"),
                                    monitor="val_loss", save_best_only=True)
]

# -----------------------------------------
# 4) Prepare targets & sample weights
# -----------------------------------------
y_train_dict = {}
y_val_dict = {}
y_test_dict = {}
sample_weights = {}

if y_rain_train is not None:
    # Ensure labels are 0/1
    y_rain_train_bin = (y_rain_train > 0.5).astype(np.float32)
    y_rain_val_bin   = (y_rain_val > 0.5).astype(np.float32)
    y_rain_test_bin  = (y_rain_test > 0.5).astype(np.float32)

    y_train_dict["rain_out"] = y_rain_train_bin
    y_val_dict["rain_out"] = y_rain_val_bin
    y_test_dict["rain_out"] = y_rain_test_bin

    # Provide dummy sample weights for rain_out (all ones)
    sw_rain = np.ones_like(y_rain_train_bin, dtype=np.float32)
    sample_weights["rain_out"] = sw_rain

if y_temp_train is not None:
    y_train_dict["temp_out"] = y_temp_train.astype(np.float32)
    y_val_dict["temp_out"] = y_temp_val.astype(np.float32)
    y_test_dict["temp_out"] = y_temp_test.astype(np.float32)

    # Mask NaNs with sample weights (0 weight where target is NaN)
    sw_temp = (~np.isnan(y_temp_train)).astype(np.float32)
    sample_weights["temp_out"] = sw_temp

if y_wind_train is not None:
    y_train_dict["wind_out"] = y_wind_train.astype(np.float32)
    y_val_dict["wind_out"] = y_wind_val.astype(np.float32)
    y_test_dict["wind_out"] = y_wind_test.astype(np.float32)

    sw_wind = (~np.isnan(y_wind_train)).astype(np.float32)
    sample_weights["wind_out"] = sw_wind

# -----------------------------------------
# 5) Train
# -----------------------------------------
print("\n🚀 Training model...")
history = model.fit(
    X_train,
    y_train_dict,
    validation_data=(X_val, y_val_dict),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    sample_weight=sample_weights if sample_weights else None,
    verbose=1
)

# -----------------------------------------
# 6) Evaluate on test set
# -----------------------------------------
print("\n📊 Evaluating on test set...")
eval_results = model.evaluate(X_test, y_test_dict, verbose=0, return_dict=True)
print("Test metrics:", eval_results)

# -----------------------------------------
# 7) Save model and metadata
# -----------------------------------------
model_path = os.path.join(MODELS_DIR, "weather_nn.keras")
model.save(model_path)

model_info = {
    "model_path": model_path,
    "created_at": datetime.utcnow().isoformat() + "Z",
    "inputs": feature_columns,
    "losses": list(losses.keys()),
    "loss_weights": loss_weights,
    "train_size": int(len(X_train)),
    "val_size": int(len(X_val)),
    "test_size": int(len(X_test)),
    "notes": "Multi-head NN with shared trunk; outputs: rain (sigmoid), temp (linear), wind (linear)."
}
with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)

with open(os.path.join(DATA_DIR, "training_metrics.json"), "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\n✅ Training complete. Model saved to: {model_path}")
print(f"ℹ️  Metrics saved to: {os.path.join(DATA_DIR, 'training_metrics.json')}")
print(f"ℹ️  Model info saved to: {os.path.join(MODELS_DIR, 'model_info.json')}")
