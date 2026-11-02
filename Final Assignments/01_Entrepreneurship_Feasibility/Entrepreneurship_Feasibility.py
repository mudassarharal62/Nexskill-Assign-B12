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

# ENTREPRENEURSHIP FEASIBILITY PREDICTION
# Target: Feasibility_Status
# Feasibility_Score and Decision_Score are excluded from predictors to reduce target leakage.

DATA_FILE = DATA_DIR / "University_Entrepreneurship_Feasibility_Dataset1.csv"
df = pd.read_csv(DATA_FILE)
show_basic_info(df, "Entrepreneurship Feasibility Dataset")

df = df.drop_duplicates().copy()
target = "Feasibility_Status"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found.")

print("\nTarget distribution:\n", df[target].value_counts(dropna=False))
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=target, order=df[target].value_counts().index)
plt.title("Feasibility Status Distribution")
plt.xticks(rotation=15)
save_plot("01_target_distribution.png")

# Numeric feature analysis
numeric_all = df.select_dtypes(include=np.number).columns.tolist()
if numeric_all:
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[numeric_all].corr(), cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation")
    save_plot("02_correlation_heatmap.png")

# Remove identifiers and columns that directly encode the outcome.
drop_cols = [target, "ID", "Feasibility_Score", "Decision_Score"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target].astype(str)
X = X.replace([np.inf, -np.inf], np.nan)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
}
results = []
best_name, best_score, best_pipeline = None, -1, None
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    score = accuracy_score(y_test, pred)
    results.append({"Model": name, "Accuracy": score})
    print(f"\n{name}\nAccuracy: {score:.4f}\n")
    print(classification_report(y_test, pred, zero_division=0))
    if score > best_score:
        best_name, best_score, best_pipeline = name, score, pipe

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df)

best_pred = best_pipeline.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, best_pred, xticks_rotation=20, cmap="Blues")
plt.title(f"Confusion Matrix: {best_name}")
save_plot("03_confusion_matrix.png")

joblib.dump(best_pipeline, MODEL_DIR / "entrepreneurship_feasibility_model.joblib")
print(f"\nBest model: {best_name}, accuracy: {best_score:.4f}")
print("Saved model and outputs in this project folder.")

# Example prediction using a real held-out row
sample = X_test.iloc[[0]]
print("\nExample prediction:", best_pipeline.predict(sample)[0])
