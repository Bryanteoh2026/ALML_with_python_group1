"""Step 3: Optional K-Means clustering for customer marketing segments."""

import os

# Limit numerical-library threads so K-Means runs consistently on student laptops.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_customer_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


sns.set_theme(style="whitegrid")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    cluster_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "NumOptionalServices",
    ]
    X_cluster = df[cluster_features]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_cluster)

    # Elbow method: compare several possible numbers of clusters.
    k_values = range(2, 9)
    inertia_values = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(scaled_features)
        inertia_values.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertia_values, marker="o")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "15_kmeans_elbow_curve.png", dpi=180)
    plt.close()

    # Four clusters provide useful marketing groups while remaining easy to explain.
    final_kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    cluster_numbers = final_kmeans.fit_predict(scaled_features)
    df["CustomerSegment"] = cluster_numbers + 1

    # Use PCA only to display the four-dimensional clusters on a 2D chart.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(scaled_features)
    plot_df = pd.DataFrame(
        {
            "PCA1": pca_values[:, 0],
            "PCA2": pca_values[:, 1],
            "CustomerSegment": df["CustomerSegment"].astype(str),
            "Churn": df["Churn"],
        }
    )

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="PCA1",
        y="PCA2",
        hue="CustomerSegment",
        alpha=0.65,
        s=45,
    )
    plt.title("Customer Segments Visualised with PCA")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "16_customer_segments_pca.png", dpi=180)
    plt.close()

    df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})
    segment_summary = (
        df.groupby("CustomerSegment")
        .agg(
            Customers=("customerID", "count"),
            Average_Tenure=("tenure", "mean"),
            Average_Monthly_Charges=("MonthlyCharges", "mean"),
            Average_Total_Charges=("TotalCharges", "mean"),
            Average_Optional_Services=("NumOptionalServices", "mean"),
            Churn_Rate=("ChurnFlag", "mean"),
        )
        .reset_index()
    )
    segment_summary.to_csv(OUTPUT_DIR / "customer_segment_summary.csv", index=False)
    df.drop(columns="ChurnFlag").to_csv(
        OUTPUT_DIR / "customers_with_segments.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=segment_summary, x="CustomerSegment", y="Churn_Rate", color="tab:blue")
    plt.title("Churn Rate by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Churn Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "17_segment_churn_rate.png", dpi=180)
    plt.close()

    print("Customer segmentation completed.")
    print(segment_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
