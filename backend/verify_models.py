
from pathlib import Path
import joblib
import xgboost
import sklearn

model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "N_TRAINED_MODELS_ALL.pkl"
)

data = joblib.load(model_path)

print("\n=== LIBRARY VERSIONS ===")
print("XGBoost:", xgboost.__version__)
print("scikit-learn:", sklearn.__version__)

print("\n=== MODEL FEATURES ===")
features = data["metadata"]["features"]

for i, feature in enumerate(features, start=1):
    print(f"{i:02d}. {repr(feature)}")

print("\n=== MODEL CONFIGURATION ===")

for name in ["lr_model", "xgb_model", "meta_learner"]:
    model = data[name]

    print(f"\n{name}")
    print("Type:", type(model).__name__)
    print("Expected inputs:", getattr(model, "n_features_in_", "Unknown"))
    print("Classes:", getattr(model, "classes_", "Unknown"))
    print("Feature names:", getattr(model, "feature_names_in_", "Not stored"))

print("\n=== METADATA ===")
print("Training samples:", data["metadata"].get("training_samples"))
print("Pakistan samples:", data["metadata"].get("pakistan_samples"))
print("Number of features:", data["metadata"].get("n_features"))

print("\nVerification complete.")
