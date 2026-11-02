# Install dependencies:
# pip install pandas numpy matplotlib seaborn scikit-learn joblib

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()

def show_basic_info(df, name):
    print(f"\n{'='*65}\n{name}\n{'='*65}")
    print("Dataset shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False).head(20))
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nDescriptive statistics:\n", df.describe(include="all").T)

# DIAMOND PRICE PREDICTION
# Target: price
# The dataset's Unnamed: 0 column is an index and is excluded from modelling.

DATA_FILE = DATA_DIR / "10-diamonds.csv"
df = pd.read_csv(DATA_FILE)
show_basic_info(df, "Diamond Dataset")

df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
df = df.drop_duplicates().copy()
target = "price"
required = {"carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required dataset columns: {missing}")

# Clean impossible dimensions and create derived features.
for col in ["x", "y", "z", "carat", "price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=[target, "carat"]).copy()
df = df[(df["carat"] > 0) & (df["price"] > 0)].copy()
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
df["volume"] = df["x"] * df["y"] * df["z"]
df["price_per_carat"] = df["price"] / df["carat"]
# price_per_carat uses the target and is intentionally NOT included as a predictor.

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=40, kde=True)
plt.title("Diamond Price Distribution")
save_plot("01_price_distribution.png")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=42), x="carat", y="price", alpha=0.5)
plt.title("Price vs Carat")
save_plot("02_price_vs_carat.png")

plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numeric Feature Correlation")
save_plot("03_correlation_heatmap.png")

X = df.drop(columns=[target, "price_per_carat"])
y = df[target]
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_features),
    ("categorical", categorical_pipe, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
}
results = []
best_name, best_rmse, best_pipeline = None, float("inf"), None
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"\n{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    if rmse < best_rmse:
        best_name, best_rmse, best_pipeline = name, rmse, pipe

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df)

pred = best_pipeline.predict(X_test)
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=pred, alpha=0.4)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted Price: {best_name}")
save_plot("04_actual_vs_predicted.png")

residuals = y_test - pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Prediction Residual Distribution")
save_plot("05_residual_distribution.png")

joblib.dump(best_pipeline, MODEL_DIR / "diamond_price_model.joblib")
pd.DataFrame({"actual_price": y_test.values, "predicted_price": pred, "absolute_error": np.abs(y_test.values-pred)}).to_csv(
    OUTPUT_DIR / "test_predictions.csv", index=False
)

print(f"\nBest model: {best_name}")
print("Saved model and outputs in this project folder.")
print("\nExample prediction:", round(float(best_pipeline.predict(X_test.iloc[[0]])[0]), 2))
