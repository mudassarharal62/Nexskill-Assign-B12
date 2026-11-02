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

# REAL ESTATE MARKET SENTIMENT ANALYSIS
# Target: sentiment_label
# The source file contains social/content records. A keyword-based domain subset is
# created for a real-estate-focused analysis when enough matching records exist.

DATA_FILE = DATA_DIR / "real_estate_market_sentiment__2026-09-18__de7a8d4a.csv"
df = pd.read_csv(DATA_FILE, low_memory=False)
show_basic_info(df, "Real Estate Market Sentiment Dataset")

df = df.drop_duplicates(subset=["id"] if "id" in df.columns else None).copy()
target = "sentiment_label"
text_col = "content"
if target not in df.columns or text_col not in df.columns:
    raise ValueError("Required columns 'sentiment_label' and 'content' are missing.")

df = df.dropna(subset=[target, text_col]).copy()
df[text_col] = df[text_col].astype(str).str.strip()
df = df[df[text_col].str.len() >= 8].copy()

# Retain records with real-estate/housing context when at least 100 records match.
terms = r"real estate|realty|housing|home price|home value|mortgage|property|rental|rent market|housing market|commercial property|residential"
domain_mask = df[text_col].str.contains(terms, case=False, regex=True, na=False)
if domain_mask.sum() >= 100:
    analysis_df = df.loc[domain_mask].copy()
    print(f"\nDomain filter: using {len(analysis_df)} real-estate-related records from {len(df)} total usable records.")
else:
    analysis_df = df.copy()
    print("\nDomain filter: not enough keyword matches, using all usable records.")

analysis_df[target] = analysis_df[target].astype(str).str.lower()
print("\nSentiment distribution:\n", analysis_df[target].value_counts())
plt.figure(figsize=(7, 5))
sns.countplot(data=analysis_df, x=target, order=analysis_df[target].value_counts().index)
plt.title("Sentiment Class Distribution")
save_plot("01_sentiment_distribution.png")

# Text length analysis
analysis_df["text_length"] = analysis_df[text_col].str.len()
plt.figure(figsize=(8, 5))
sns.histplot(analysis_df["text_length"], bins=30, kde=True)
plt.title("Content Length Distribution")
save_plot("02_text_length_distribution.png")

# Optional date-based trend using published_at.
if "published_at" in analysis_df.columns:
    analysis_df["published_at"] = pd.to_datetime(analysis_df["published_at"], errors="coerce", utc=True)
    dated = analysis_df.dropna(subset=["published_at"]).copy()
    if not dated.empty:
        dated["month"] = dated["published_at"].dt.to_period("M").astype(str)
        trend = dated.groupby(["month", target]).size().unstack(fill_value=0)
        trend.to_csv(OUTPUT_DIR / "monthly_sentiment_counts.csv")
        trend.plot(figsize=(12, 5), marker="o")
        plt.title("Monthly Sentiment Counts")
        plt.xlabel("Month")
        plt.ylabel("Number of records")
        save_plot("03_monthly_sentiment_trend.png")

X = analysis_df[text_col].astype(str)
y = analysis_df[target]
# Stratify only when every class has at least two records.
stratify_y = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=stratify_y
)

# TF-IDF + Logistic Regression baseline, suitable for labelled text data.
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        max_features=30000, min_df=2, sublinear_tf=True
    )),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("\nTF-IDF + Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=25, cmap="Blues")
plt.title("Sentiment Model Confusion Matrix")
save_plot("04_confusion_matrix.png")

joblib.dump(model, MODEL_DIR / "real_estate_sentiment_model.joblib")
pd.DataFrame({"actual": y_test.values, "predicted": pred, "content": X_test.values}).to_csv(
    OUTPUT_DIR / "test_predictions.csv", index=False
)

# Most informative terms from the fitted linear classifier.
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]
feature_names = np.array(vectorizer.get_feature_names_out())
rows = []
for i, label in enumerate(classifier.classes_):
    top_ids = np.argsort(classifier.coef_[i])[-15:][::-1]
    for rank, idx in enumerate(top_ids, start=1):
        rows.append({"sentiment": label, "rank": rank, "term": feature_names[idx], "weight": classifier.coef_[i][idx]})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "important_sentiment_terms.csv", index=False)

print("\nSaved model and outputs. Note: sentiment labels are inherited from the supplied dataset, not independently verified.")
