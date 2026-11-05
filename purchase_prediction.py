# ============================================================
# CUSTOMER PURCHASE PREDICTION
# DATASET CREATION + MACHINE LEARNING MODEL TRAINING
# ============================================================

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==================== 1. FILE PATHS ====================

PROJECT_DIR = Path(
    r"C:\Users\acer\PycharmProjects\PythonProject15"
)

DATA_DIR = PROJECT_DIR / "datas"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "cleaned_online_retail.csv"

OUTPUT_FILE = DATA_DIR / "customer_purchase_prediction_data.csv"

print("=" * 60)
print("CUSTOMER PURCHASE PREDICTION")
print("=" * 60)

print("Looking for:", DATA_FILE)
print("File exists:", DATA_FILE.exists())

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Check the file path."
    )

# ==================== 2. LOAD DATA ====================

df = pd.read_csv(DATA_FILE, low_memory=False)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["Customer ID"] = pd.to_numeric(
    df["Customer ID"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)

df = df.dropna(
    subset=["Customer ID", "InvoiceDate", "Invoice"]
).copy()

# Remove unknown customers
df = df[df["Customer ID"] != -1].copy()

# Keep only valid sales
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

df["Customer ID"] = df["Customer ID"].astype(int).astype(str)

df["TotalSales"] = df["Quantity"] * df["Price"]

print("\nDataset loaded successfully!")
print("Rows:", len(df))

# ==================== 3. DEFINE TIME PERIODS ====================

max_date = df["InvoiceDate"].max()

cutoff_date = max_date - pd.Timedelta(days=90)

print("\nLatest Date:", max_date)
print("Prediction Cutoff Date:", cutoff_date)

# Historical customer data
history = df[
    df["InvoiceDate"] <= cutoff_date
].copy()

# Purchases during the following 90 days
future = df[
    (df["InvoiceDate"] > cutoff_date) &
    (df["InvoiceDate"] <= max_date)
].copy()

if history.empty or future.empty:
    raise ValueError(
        "Not enough historical or future data for 90-day prediction."
    )

# ==================== 4. CREATE CUSTOMER FEATURES ====================

features = history.groupby("Customer ID").agg(
    Recency=(
        "InvoiceDate",
        lambda x: (cutoff_date - x.max()).days
    ),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalSales", "sum")
)

order_counts = history.groupby("Customer ID")[
    "Invoice"
].nunique()

features["AverageOrderValue"] = (
    features["Monetary"] / order_counts.clip(lower=1)
)

# ==================== 5. CREATE TARGET VARIABLE ====================

future_customers = set(
    future["Customer ID"].unique()
)

features["RepeatPurchase90Days"] = (
    features.index.isin(future_customers).astype(int)
)

features = features.reset_index()

# Save prediction dataset
features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPurchase prediction dataset created!")
print("Total Customers:", len(features))
print("\nTarget Distribution:")
print(features["RepeatPurchase90Days"].value_counts())

print("\nDataset saved:", OUTPUT_FILE)

# ==================== 6. PREPARE TRAINING DATA ====================

X = features[
    [
        "Recency",
        "Frequency",
        "Monetary",
        "AverageOrderValue"
    ]
]

y = features["RepeatPurchase90Days"]

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Both target classes need at least 2 customers. "
        "Check the 90-day target distribution."
    )

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Customers:", len(X_train))
print("Testing Customers:", len(X_test))

# ==================== 7. LOGISTIC REGRESSION ====================

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
)

logistic_model.fit(X_train, y_train)

lr_predictions = logistic_model.predict(X_test)

lr_probabilities = logistic_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print("Accuracy:",
      round(accuracy_score(y_test, lr_predictions), 4))

print("Precision:",
      round(precision_score(y_test, lr_predictions, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, lr_predictions, zero_division=0), 4))

print("F1-Score:",
      round(f1_score(y_test, lr_predictions, zero_division=0), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, lr_probabilities), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    lr_predictions,
    zero_division=0
))

# ==================== 8. RANDOM FOREST ====================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print("Accuracy:",
      round(accuracy_score(y_test, rf_predictions), 4))

print("Precision:",
      round(precision_score(y_test, rf_predictions, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, rf_predictions, zero_division=0), 4))

print("F1-Score:",
      round(f1_score(y_test, rf_predictions, zero_division=0), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, rf_probabilities), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    zero_division=0
))

# ==================== 9. FEATURE IMPORTANCE ====================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    DATA_DIR / "purchase_prediction_feature_importance.csv",
    index=False
)

# ==================== 10. FINAL SUMMARY ====================

print("\n" + "=" * 60)
print("PURCHASE PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("1. Purchase prediction dataset created.")
print("2. Logistic Regression model trained.")
print("3. Random Forest model trained.")
print("4. Accuracy and ROC-AUC evaluated.")
print("5. Feature importance calculated.")

print("\nSaved Files:")
print("1. customer_purchase_prediction_data.csv")
print("2. purchase_prediction_feature_importance.csv")