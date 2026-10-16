
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" /
    "colorectal_cancer_dataset_1_kaggle (2).csv"
)

MODEL_PATH = (
    ROOT / "backend" / "models" /
    "N_TRAINED_MODELS_ALL.pkl"
)

OUTPUT_PATH = (
    ROOT / "backend" / "models" /
    "recovered_preprocessing.joblib"
)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Loading model metadata...")
models = joblib.load(MODEL_PATH)
features = models["metadata"]["features"]

global_df = df[df["Country"] != "Pakistan"].copy()
pakistan_df = df[df["Country"] == "Pakistan"].copy()

X_global = global_df[features].copy()

# Reproduce the original research target definition.
y_global = global_df["Cancer_Stage"].apply(
    lambda x: 1 if x in ["Regional", "Distant"] else 0
)

X_train, X_test, y_train, y_test = train_test_split(
    X_global,
    y_global,
    test_size=0.2,
    stratify=y_global,
    random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Pakistan rows:", len(pakistan_df))

assert len(X_train) == models["metadata"]["training_samples"]
assert len(pakistan_df) == models["metadata"]["pakistan_samples"]

# Reconstruct categorical encoders.
categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

encoders = {}

for col in categorical_cols:
    encoder = LabelEncoder()
    X_train[col] = encoder.fit_transform(
        X_train[col].astype(str)
    )
    encoders[col] = encoder

# Reconstruct imputer and scaler.
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

preprocessing = {
    "features": features,
    "categorical_cols": categorical_cols,
    "encoders": encoders,
    "imputer": imputer,
    "scaler": scaler,
    "training_rows": len(X_train),
    "random_seed": 42,
}

joblib.dump(preprocessing, OUTPUT_PATH)

print("\n=== PREPROCESSING RECOVERED ===")
print("Features:", len(features))
print("Categorical columns:", categorical_cols)
print("Scaler input features:", scaler.n_features_in_)
print("Output:", OUTPUT_PATH)
print("Recovery complete.")
