
from pathlib import Path
import sys

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

sys.path.insert(0, str(BASE_DIR))

from app import app

prep = joblib.load(
    BASE_DIR / "models" / "recovered_preprocessing.joblib"
)

features = prep["features"]

df = pd.read_csv(
    ROOT_DIR / "data" /
    "colorectal_cancer_dataset_1_kaggle (2).csv",
    nrows=1
)

row = df[features].iloc[0]

payload = {
    key: (
        None if pd.isna(value)
        else value.item() if hasattr(value, "item")
        else value
    )
    for key, value in row.items()
}

print("\n=== DIRECT FLASK API TEST ===")
print("Number of features:", len(payload))

client = app.test_client()

response = client.post(
    "/predict",
    json=payload
)

print("HTTP Status:", response.status_code)
print("Response:", response.get_json())

if response.status_code == 200:
    print("\nSUCCESS: Flask prediction endpoint works.")
else:
    print("\nFAILED: Check Flask error details.")
