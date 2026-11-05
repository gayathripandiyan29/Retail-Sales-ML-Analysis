# ============================================================
# CUSTOMER SEGMENTATION - RFM + K-MEANS + BUSINESS INSIGHTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==================== 1. FILE PATHS ====================

PROJECT_DIR = Path(
    r"C:\Users\acer\PycharmProjects\PythonProject15"
)

DATA_DIR = PROJECT_DIR / "datas"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = DATA_DIR / "cleaned_online_retail.csv"

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {FILE_PATH}"
    )

print("=" * 60)
print("CUSTOMER SEGMENTATION - RFM + K-MEANS")
print("=" * 60)

# ==================== 2. LOAD DATA ====================

df = pd.read_csv(FILE_PATH, low_memory=False)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["Customer ID"] = pd.to_numeric(
    df["Customer ID"], errors="coerce"
)

df["TotalSales"] = pd.to_numeric(
    df["TotalSales"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df = df.dropna(
    subset=["InvoiceDate", "Customer ID", "TotalSales"]
).copy()

# Exclude unknown customers
df = df[df["Customer ID"] != -1].copy()

# Keep valid sales only
df = df[
    (df["TotalSales"] > 0) &
    (df["Quantity"] > 0)
].copy()

if df.empty:
    raise ValueError("No valid customer records found.")

print("Dataset loaded:", df.shape)
print("Unique customers:", df["Customer ID"].nunique())

# ==================== 3. CREATE RFM FEATURES ====================

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    Recency=(
        "InvoiceDate",
        lambda x: (reference_date - x.max()).days
    ),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalSales", "sum")
).reset_index()

rfm["AvgOrderValue"] = (
    rfm["Monetary"] / rfm["Frequency"].clip(lower=1)
)

print("\n========== RFM SUMMARY ==========")
print(
    rfm[["Recency", "Frequency", "Monetary"]].describe()
)

# ==================== 4. RFM RULE-BASED SEGMENTATION ====================

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


def assign_rfm_segment(row):

    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r >= 4 and f >= 2:
        return "Potential Loyalists"

    elif r <= 2 and f >= 3:
        return "At Risk"

    else:
        return "Needs Attention"


rfm["RFM_Segment"] = rfm.apply(
    assign_rfm_segment,
    axis=1
)

# ==================== 5. PREPARE K-MEANS DATA ====================

features = rfm[
    ["Recency", "Frequency", "Monetary"]
].copy()

# Reduce skewness
features = np.log1p(features)

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# ==================== 6. ELBOW METHOD ====================

inertia_values = []
k_values = range(2, 8)

for k in k_values:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(scaled_features)

    inertia_values.append(km.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("K-Means Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    DATA_DIR / "kmeans_elbow.png",
    dpi=150
)

plt.close()

# ==================== 7. SILHOUETTE SCORE ====================

silhouette_results = []

for k in range(2, 8):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(scaled_features)

    score = silhouette_score(
        scaled_features,
        labels
    )

    silhouette_results.append({
        "K": k,
        "Silhouette_Score": score
    })

silhouette_df = pd.DataFrame(silhouette_results)

best_k = int(
    silhouette_df.loc[
        silhouette_df["Silhouette_Score"].idxmax(),
        "K"
    ]
)

print("\n========== SILHOUETTE RESULTS ==========")
print(silhouette_df.to_string(index=False))
print("\nSelected K:", best_k)

plt.figure(figsize=(8, 5))

plt.plot(
    silhouette_df["K"],
    silhouette_df["Silhouette_Score"],
    marker="o"
)

plt.title("K-Means Silhouette Score")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    DATA_DIR / "kmeans_silhouette.png",
    dpi=150
)

plt.close()

# ==================== 8. FINAL K-MEANS MODEL ====================

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = final_model.fit_predict(
    scaled_features
)

# ==================== 9. CLUSTER SUMMARY ====================

cluster_summary = rfm.groupby("Cluster").agg(
    Customers=("Customer ID", "count"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    AvgOrderValue=("AvgOrderValue", "mean")
).reset_index()

# Higher frequency and monetary are better.
# Lower recency is better.
cluster_summary["Value_Score"] = (
    cluster_summary["AvgFrequency"].rank(pct=True) +
    cluster_summary["AvgMonetary"].rank(pct=True) +
    (1 - cluster_summary["AvgRecency"].rank(pct=True))
)

# Identify the most valuable cluster
loyal_cluster = cluster_summary.loc[
    cluster_summary["Value_Score"].idxmax(),
    "Cluster"
]

remaining = cluster_summary[
    cluster_summary["Cluster"] != loyal_cluster
].copy()

# Identify the least active remaining cluster
remaining["Risk_Score"] = (
    remaining["AvgRecency"].rank(pct=True) -
    remaining["AvgFrequency"].rank(pct=True) -
    remaining["AvgMonetary"].rank(pct=True)
)

at_risk_cluster = remaining.loc[
    remaining["Risk_Score"].idxmax(),
    "Cluster"
]

# Assign business-friendly names
cluster_names = {}

for cluster_id in cluster_summary["Cluster"]:

    if cluster_id == loyal_cluster:
        cluster_names[cluster_id] = "Loyal Customers"

    elif cluster_id == at_risk_cluster:
        cluster_names[cluster_id] = "At-Risk Customers"

    else:
        cluster_names[cluster_id] = "Potential Customers"


rfm["KMeans_Segment"] = rfm["Cluster"].map(
    cluster_names
)

cluster_summary["Segment_Name"] = (
    cluster_summary["Cluster"].map(cluster_names)
)

cluster_summary = cluster_summary.drop(
    columns=["Value_Score"]
)

# ==================== 10. COMPARE RFM AND K-MEANS ====================

comparison = pd.crosstab(
    rfm["RFM_Segment"],
    rfm["KMeans_Segment"],
    margins=True
)

print("\n========== RFM VS K-MEANS COMPARISON ==========")
print(comparison)

comparison.to_csv(
    DATA_DIR / "rfm_vs_kmeans_comparison.csv"
)

# ==================== 11. SAVE OUTPUT FILES ====================

rfm.to_csv(
    DATA_DIR / "customer_kmeans_segments.csv",
    index=False
)

cluster_summary.to_csv(
    DATA_DIR / "kmeans_cluster_summary.csv",
    index=False
)

silhouette_df.to_csv(
    DATA_DIR / "kmeans_silhouette_scores.csv",
    index=False
)

# ==================== 12. BUSINESS INSIGHTS ====================

print("\n========== CUSTOMER CLUSTER SUMMARY ==========")

print(
    cluster_summary[
        [
            "Cluster",
            "Segment_Name",
            "Customers",
            "AvgRecency",
            "AvgFrequency",
            "AvgMonetary"
        ]
    ].round(2).to_string(index=False)
)

print("\n========== BUSINESS INSIGHTS ==========")

print(
    "1. Loyal Customers: "
    "Provide loyalty rewards and exclusive offers."
)

print(
    "2. At-Risk Customers: "
    "Send special discounts to encourage repeat purchases."
)

print(
    "3. Potential Customers: "
    "Use welcome offers and engagement campaigns."
)

print("\nRFM Segment Counts:")
print(rfm["RFM_Segment"].value_counts())

print("\nK-Means Segment Counts:")
print(rfm["KMeans_Segment"].value_counts())

# ==================== 13. FINAL OUTPUT ====================

print("\n========== FILES SAVED ==========")

print("1. customer_kmeans_segments.csv")
print("2. kmeans_cluster_summary.csv")
print("3. rfm_vs_kmeans_comparison.csv")
print("4. kmeans_elbow.png")
print("5. kmeans_silhouette.png")
print("6. kmeans_silhouette_scores.csv")

print("\nCUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")