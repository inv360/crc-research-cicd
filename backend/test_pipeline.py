
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "backend" / "models"

models = joblib.load(
    MODEL_DIR / "N_TRAINED_MODELS_ALL.pkl"
)

prep = joblib.load(
    MODEL_DIR / "recovered_preprocessing.joblib"
)

features = prep["features"]

df = pd.read_csv(
    ROOT / "data" /
    "colorectal_cancer_dataset_1_kaggle (2).csv"
)

# Use one existing record only for technical verification.
row = df.loc[
    df["Country"] != "Pakistan",
    features
].head(1).copy()

print("Original row selected.")
print("Number of features:", len(row.columns))

# Apply recovered categorical encoders.
for col in prep["categorical_cols"]:
    encoder = prep["encoders"][col]
    values = row[col].astype(str)

    unknown = set(values) - set(encoder.classes_)
    if unknown:
        raise ValueError(
            f"Unknown category in {col}: {unknown}"
        )

    row[col] = encoder.transform(values)

# Apply recovered imputer and scaler.
X_imputed = prep["imputer"].transform(row)
X_scaled = prep["scaler"].transform(X_imputed)

print("Scaled shape:", X_scaled.shape)
print("All values finite:", np.isfinite(X_scaled).all())

# Base-model probabilities.
lr_probability = models["lr_model"].predict_proba(
    X_scaled
)[:, 1]

xgb_probability = models["xgb_model"].predict_proba(
    X_scaled
)[:, 1]

# SHAP magnitude.
explainer = shap.TreeExplainer(models["xgb_model"])
shap_values = explainer.shap_values(X_scaled)

shap_magnitude = np.abs(shap_values).mean(axis=1)

# Original stacking formula.
X_meta = np.column_stack([
    lr_probability,
    xgb_probability,
    shap_magnitude
])

svse_probability = models["meta_learner"].predict_proba(
    X_meta
)[:, 1]

print("\n=== PIPELINE SMOKE TEST ===")
print("LR probability:", float(lr_probability[0]))
print("XGBoost probability:", float(xgb_probability[0]))
print("SHAP magnitude:", float(shap_magnitude[0]))
print("Meta-features shape:", X_meta.shape)
print("SVSE probability:", float(svse_probability[0]))
print("Technical smoke test complete.")
