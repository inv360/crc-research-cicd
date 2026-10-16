
from pathlib import Path
import joblib
import shap

model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "N_TRAINED_MODELS_ALL.pkl"
)

models = joblib.load(model_path)
xgb_model = models["xgb_model"]

print("Creating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(xgb_model)

print("SHAP TreeExplainer created successfully.")
print("Model expects:", xgb_model.n_features_in_, "features")
