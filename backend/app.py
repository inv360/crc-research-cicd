
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/predict": {
            "origins": ["http://localhost:4200"]
        }
    }
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

print("Loading trained CRC research models...")

models = joblib.load(
    MODEL_DIR / "N_TRAINED_MODELS_ALL.pkl"
)

prep = joblib.load(
    MODEL_DIR / "recovered_preprocessing.joblib"
)

FEATURES = prep["features"]

lr_model = models["lr_model"]
xgb_model = models["xgb_model"]
meta_learner = models["meta_learner"]

explainer = shap.TreeExplainer(xgb_model)

print("Models and preprocessing loaded successfully.")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "application": "CRC Research Prediction API",
        "models_loaded": True,
        "features": len(FEATURES)
    })


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Expected a JSON object."
            }), 400

        missing = [
            feature for feature in FEATURES
            if feature not in payload
        ]

        if missing:
            return jsonify({
                "error": "Missing required features.",
                "missing": missing
            }), 400

        row = pd.DataFrame(
            [{f: payload[f] for f in FEATURES}],
            columns=FEATURES
        )

        # Encode categorical inputs.
        for col in prep["categorical_cols"]:
            encoder = prep["encoders"][col]

            values = row[col].astype(str)
            unknown = set(values) - set(encoder.classes_)

            if unknown:
                return jsonify({
                    "error": f"Unknown category for {col}",
                    "allowed_values": encoder.classes_.tolist()
                }), 400

            row[col] = encoder.transform(values)

        # Validate numeric inputs.
        for col in FEATURES:
            if col not in prep["categorical_cols"]:
                row[col] = pd.to_numeric(
                    row[col], errors="coerce"
                )

        # Apply original preprocessing sequence.
        X_imputed = prep["imputer"].transform(row)
        X_scaled = prep["scaler"].transform(X_imputed)

        if not np.isfinite(X_scaled).all():
            return jsonify({
                "error": "Invalid numerical input."
            }), 400

        # Base model probabilities.
        lr_prob = lr_model.predict_proba(
            X_scaled
        )[:, 1]

        xgb_prob = xgb_model.predict_proba(
            X_scaled
        )[:, 1]

        # SHAP-based meta feature.
        shap_values = explainer.shap_values(X_scaled)

        shap_magnitude = np.abs(
            shap_values
        ).mean(axis=1)

        X_meta = np.column_stack([
            lr_prob,
            xgb_prob,
            shap_magnitude
        ])

        final_prob = meta_learner.predict_proba(
            X_meta
        )[:, 1]

        return jsonify({
            "model": "SVSE Research Ensemble",
            "probability": float(final_prob[0]),
            "lr_probability": float(lr_prob[0]),
            "xgb_probability": float(xgb_prob[0]),
            "shap_magnitude": float(shap_magnitude[0]),
            "disclaimer": (
                "Research prototype only. "
                "Not validated for clinical use."
            )
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction processing failed."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
