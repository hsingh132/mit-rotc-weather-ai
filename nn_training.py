"""
nn_training.py
Master control script to prepare data and train the neural network model.

This script runs:
1. prepare_training_data.py  - builds the training-ready dataset
2. train_model.py            - trains the multi-output neural network

Run from the project root:
    python3 nn_training.py
"""

import subprocess
import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "training_scripts")

prepare_script = os.path.join(SCRIPTS_DIR, "prepare_training_data.py")
train_script = os.path.join(SCRIPTS_DIR, "train_model.py")

# === Execution ===
print("🧠 Starting neural network training pipeline...\n")

print("[1/2] Preparing training data...")
subprocess.run(["python3", prepare_script], check=True)
print("✅ Data preparation complete.\n")

print("[2/2] Training neural network model...")
subprocess.run(["python3", train_script], check=True)
print("✅ Model training complete.\n")

print("🎯 Neural network pipeline finished successfully!")
