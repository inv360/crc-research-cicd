from pathlib import Path
import joblib

model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "N_TRAINED_MODELS_ALL.pkl"
)

print("Model file:", model_path)
print("File exists:", model_path.exists())

if not model_path.exists():
    raise FileNotFoundError(
        "Model file not found. Check backend/models folder."
    )

data = joblib.load(model_path)

print("\nLoaded object type:", type(data))

if isinstance(data, dict):
    print("\nAvailable model keys:")

    for key, value in data.items():
        print(f"{key}: {type(value)}")

    if "metadata" in data:
        print("\nModel metadata:")
        print(data["metadata"])

else:
    print("\nLoaded model:")
    print(data)

    if hasattr(data, "feature_names_in_"):
        print("\nFeature names:")
        print(data.feature_names_in_)