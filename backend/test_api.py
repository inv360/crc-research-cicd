
from pathlib import Path

import joblib
import pandas as pd
import requests

# ============================================================
# STEP 1: Define project paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PREPROCESSING_PATH = (
    ROOT / "backend" / "models" /
    "recovered_preprocessing.joblib"
)

DATA_PATH = (
    ROOT / "data" /
    "colorectal_cancer_dataset_1_kaggle (2).csv"
)

API_URL = "http://127.0.0.1:5000/predict"

# ============================================================
# STEP 2: Load the original 24-feature input schema
# ============================================================

prep = joblib.load(PREPROCESSING_PATH)
features = prep["features"]

# ============================================================
# STEP 3: Select one sample record
# ============================================================

df = pd.read_csv(DATA_PATH, nrows=1)
row = df[features].iloc[0]

# ============================================================
# STEP 4: Convert values into JSON-compatible Python types
# ============================================================

payload = {}

for key, value in row.items():

    if pd.isna(value):
        payload[key] = None

    elif hasattr(value, "item"):
        payload[key] = value.item()

    else:
        payload[key] = value

print("\n=== CRC RESEARCH API TEST ===")
print("Number of features:", len(payload))

assert len(payload) == 24, "Expected exactly 24 features."

# ============================================================
# STEP 5: Create a direct HTTP session
# ============================================================

# Do not inherit environment proxy settings.
# This is the configuration that passed our previous test.

session = requests.Session()
session.trust_env = False

# ============================================================
# STEP 6: Send POST request to Flask
# ============================================================

print("\nSending prediction request to Flask...")

try:

    response = session.post(
        API_URL,
        json=payload,
        timeout=60
    )

    print("\nHTTP Status:", response.status_code)

    print("\n=== FLASK PREDICTION RESPONSE ===")

    response.raise_for_status()

    result = response.json()

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== FINAL SVSE ENSEMBLE OUTPUT ===")

    probability = result["probability"]

    print(f"Positive-class model score: {probability:.6f}")
    print(f"Percentage representation: {probability * 100:.2f}%")

    print(
        "\nDisclaimer: Research prototype only. "
        "Not validated for clinical use."
    )

    print("\nSUCCESS: API test completed successfully.")

except requests.exceptions.RequestException as error:

    print("\nERROR: HTTP request failed.")
    print(error)

    if "response" in locals():
        print("\nServer response:")
        print(response.text[:1000])

except (ValueError, KeyError) as error:

    print("\nERROR: Invalid or incomplete API response.")
    print(error)

finally:

    session.close()
